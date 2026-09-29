//! Adapt the shared HEP integral family to the private one-loop recurrence chart.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
};

use feynkit_graph::IntegralFamily as SharedFamily;
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    symbol,
};

use crate::{
    OneLoopError,
    recurrence::RecurrenceInput,
    reduce::{MAX_TOTAL_INDEX, Reduction, reduce},
    symbols::{S, scalar_product, validate_namespace},
};

fn invalid(message: impl ToString) -> OneLoopError {
    OneLoopError::InvalidFamily(message.to_string())
}

/// Reduce a shared family using signed denominator powers and a scalar numerator.
///
/// No integral-family or scalar-product representation is exposed by this
/// adapter. A symbolic dimension is required so Laurent multiplication retains
/// the finite contributions from dimension-dependent reduction coefficients.
pub fn reduce_family(
    family: &SharedFamily,
    powers: &[i32],
    numerator: &Atom,
) -> Result<Reduction, OneLoopError> {
    if family.loop_momenta().len() != 1 {
        return Err(OneLoopError::UnsupportedLoopOrder {
            found: family.loop_momenta().len(),
        });
    }
    if powers.len() != family.denominators().len() {
        return Err(invalid(
            "supply one signed power per shared-family denominator",
        ));
    }
    let dimension = family.kinematics().dimension().to_symbolic();
    if !matches!(dimension.as_view(), AtomView::Var(_)) {
        return Err(invalid(
            "one-loop reduction requires a symbolic Kinematics dimension; use Kinematics(S(\"D\")) to retain epsilon-dependent finite terms",
        ));
    }
    for input in std::iter::once(numerator)
        .chain(family.denominators())
        .chain(family.loop_momenta())
        .chain(family.external_momenta())
        .chain(std::iter::once(&dimension))
    {
        validate_namespace(input)?;
    }
    // Bound the work before partial fractions or negative powers expand. Pure
    // scalar denominators are coefficients and do not consume these bounds.
    let mut positive_total = 0i64;
    for (denominator, power) in family.denominators().iter().zip(powers) {
        if contains_loop(denominator, &family.loop_momenta()[0]) {
            if *power < -20 {
                return Err(invalid(
                    "negative denominator power exceeds supported numerator degree 20",
                ));
            }
            positive_total += i64::from((*power).max(0));
        }
    }
    if positive_total > i64::from(MAX_TOTAL_INDEX) {
        return Err(OneLoopError::UnsupportedIndex {
            found: powers.to_vec(),
            max: MAX_TOTAL_INDEX,
        });
    }
    // Validate before dropping a scaleless sector, including hidden loop
    // dependence inside arbitrary functions or uncontracted tensors.
    validate_numerator(family, &family.kinematics().apply(numerator))?;
    let mut hygiene = ChartSymbols::new(family, numerator)?;
    let mut result = Vec::new();
    for (coefficient, powers) in family.partial_fraction(powers, 100_000).map_err(invalid)? {
        let reduction = reduce_sector(family, &powers, numerator, &mut hygiene)?;
        let coefficient = hygiene.protect(&coefficient, &[]);
        result.extend(reduction.terms.into_iter().map(|(value, master)| {
            let value = (coefficient.clone() * value)
                .replace(Atom::var(S.d))
                .with(dimension.clone());
            (
                hygiene.restore(&value),
                master.map_arguments(|arg| hygiene.restore(arg)),
            )
        }));
    }
    Ok(Reduction { terms: result }.simplify())
}

/// Keep user parameters independent from recurrence polynomial coordinates.
/// These substitutions never change the shared family or its assumptions.
struct ChartSymbols {
    nonce: u64,
    used: BTreeSet<Symbol>,
    forward: BTreeMap<Symbol, Atom>,
    backward: BTreeMap<Symbol, Atom>,
}

impl ChartSymbols {
    fn new(family: &SharedFamily, numerator: &Atom) -> Result<Self, OneLoopError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut used = BTreeSet::new();
        let mut collect = |expression: &Atom| {
            expression.visitor(&mut |view| {
                if let Some(symbol) = view.get_symbol() {
                    used.insert(symbol);
                }
                true
            });
        };
        for expression in std::iter::once(numerator)
            .chain(family.denominators())
            .chain(family.loop_momenta())
            .chain(family.external_momenta())
        {
            collect(expression);
        }
        collect(&family.kinematics().apply(numerator));
        collect(&family.kinematics().dimension().to_symbolic());
        // Assumptions can introduce scalar parameters absent from denominators.
        for left in family.external_momenta() {
            for right in family.external_momenta() {
                collect(
                    &family
                        .kinematics()
                        .scalar_product(left, right)
                        .map_err(invalid)?,
                );
            }
        }
        Ok(Self {
            nonce: NEXT.fetch_add(1, Ordering::Relaxed),
            used,
            forward: BTreeMap::new(),
            backward: BTreeMap::new(),
        })
    }

    fn protect(&mut self, expression: &Atom, chart_products: &[Atom]) -> Atom {
        expression.replace_map(|view, _, output| {
            if chart_products
                .iter()
                .any(|product| product.as_view() == view)
            {
                // An identical output stops traversal into the Spenso scalar
                // product, whose dimension belongs to the recurrence.
                **output = view.to_owned();
            } else if let AtomView::Var(variable) = view {
                let symbol = variable.get_symbol();
                let Some(name) = symbol.get_name().strip_prefix("oneloopmaster::") else {
                    return;
                };
                let indexed = |prefix: &str| {
                    name.strip_prefix(prefix).is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_digit())
                    })
                };
                if !matches!(name, "d" | "k" | "xll" | "reg_delta")
                    && !indexed("q")
                    && !indexed("xq")
                    && !indexed("den")
                {
                    return;
                }
                let replacement = if let Some(replacement) = self.forward.get(&symbol) {
                    replacement.clone()
                } else {
                    let mut index = self.forward.len();
                    let placeholder = loop {
                        let candidate =
                            symbol!(format!("oneloopmaster::__input_{}_{}", self.nonce, index));
                        if self.used.insert(candidate) {
                            break candidate;
                        }
                        index += 1;
                    };
                    let replacement = Atom::var(placeholder);
                    self.forward.insert(symbol, replacement.clone());
                    self.backward.insert(placeholder, Atom::var(symbol));
                    replacement
                };
                **output = replacement;
            }
        })
    }

    fn restore(&self, expression: &Atom) -> Atom {
        expression.replace_map(|view, _, output| {
            if let AtomView::Var(variable) = view
                && let Some(original) = self.backward.get(&variable.get_symbol())
            {
                **output = original.clone();
            }
        })
    }
}

fn contains_loop(expression: &Atom, momentum: &Atom) -> bool {
    let mut found = false;
    expression.visitor(&mut |view| {
        found |= match (momentum.as_view(), view) {
            (AtomView::Var(p), _) => view.get_symbol() == Some(p.get_symbol()),
            (AtomView::Fun(p), AtomView::Fun(candidate)) => {
                p.get_symbol() == candidate.get_symbol()
                    && candidate.get_nargs() >= p.get_nargs()
                    && p.iter().zip(candidate.iter()).all(|(a, b)| a == b)
            }
            _ => false,
        };
        !found
    });
    found
}

fn monomial_powers(key: AtomView<'_>, variables: &[Atom]) -> Result<Vec<i32>, OneLoopError> {
    let mut powers = vec![0; variables.len()];
    let factors = match key {
        AtomView::Mul(mul) => mul.iter().collect::<Vec<_>>(),
        _ if key.is_one() => return Ok(powers),
        _ => vec![key],
    };
    for factor in factors {
        let (base, exponent) = match factor {
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                (
                    base,
                    i32::try_from(exponent).map_err(|_| {
                        invalid("numerator must be polynomial in the shared loop scalar products")
                    })?,
                )
            }
            _ => (factor, 1),
        };
        let index = variables
            .iter()
            .position(|variable| variable.as_view() == base)
            .ok_or_else(|| {
                invalid("numerator must be polynomial in the shared loop scalar products")
            })?;
        if !(0..=20).contains(&exponent) {
            return Err(invalid(
                "numerator degree in each loop scalar product must be between zero and 20",
            ));
        }
        powers[index] += exponent;
    }
    Ok(powers)
}

fn validate_numerator(family: &SharedFamily, numerator: &Atom) -> Result<(), OneLoopError> {
    for (key, coefficient) in numerator.coefficient_list::<i32>(family.scalar_products()) {
        monomial_powers(key.as_view(), family.scalar_products())?;
        if contains_loop(&coefficient, &family.loop_momenta()[0]) {
            return Err(invalid(
                "numerator contains unresolved loop momentum; contract tensors and use Kinematics.scalar_product",
            ));
        }
    }
    Ok(())
}

fn rank(expressions: &[Atom], variables: &[Atom]) -> Result<usize, OneLoopError> {
    if expressions.is_empty() || variables.is_empty() {
        return Ok(0);
    }
    Atom::system_to_matrix::<u16, _, _>(expressions, variables)
        .map(|(matrix, _)| matrix.rank())
        .map_err(invalid)
}

fn reduce_sector(
    family: &SharedFamily,
    powers: &[i32],
    numerator: &Atom,
    hygiene: &mut ChartSymbols,
) -> Result<Reduction, OneLoopError> {
    let kin = family.kinematics();
    let external = family.external_momenta();
    let loop_momentum = &family.loop_momenta()[0];
    let mut prefactor = Atom::num(1);
    let mut numerator = kin.apply(numerator);
    let mut shifts = Vec::new();
    let mut masses = Vec::new();
    let mut exponents = Vec::new();
    for (index, power) in powers.iter().copied().enumerate() {
        if power < 0 {
            let degree = power
                .checked_neg()
                .ok_or_else(|| invalid("negative numerator power is too large"))?;
            numerator *= family.denominators()[index].pow(degree);
        } else if power > 0 {
            let Some(quadratic) = family.quadratic_denominator(index).map_err(invalid)? else {
                if contains_loop(&family.denominators()[index], loop_momentum) {
                    return Err(invalid(
                        "positive-power eikonal denominators are not supported by one-loop master reduction",
                    ));
                }
                prefactor *= family.denominators()[index].pow(-power);
                continue;
            };
            let shift = (quadratic.momentum() - loop_momentum).expand();
            for (_, coefficient) in shift.coefficient_list::<i32>(external) {
                if !coefficient.is_real().is_true() {
                    return Err(invalid(
                        "quadratic propagator shifts must have real coefficients",
                    ));
                }
            }
            prefactor *= quadratic.scale().pow(-power);
            shifts.push(shift);
            masses.push((-quadratic.remainder() / quadratic.scale()).cancel());
            exponents.push(power);
        }
    }
    validate_numerator(family, &numerator)?;
    if shifts.is_empty() {
        return Ok(Reduction { terms: Vec::new() });
    }
    let origin = shifts[0].clone();
    for shift in &mut shifts {
        *shift = (&*shift - &origin).expand();
    }
    let mut directions = shifts
        .windows(2)
        .map(|p| (&p[1] - &p[0]).expand())
        .collect::<Vec<_>>();
    // Extra numerator directions must retain their own Gram products. Add
    // zero-power auxiliary lines only to supply those coordinates; they never
    // introduce extra denominators into the represented integral.
    let mut current_rank = rank(&directions, external)?;
    for momentum in external {
        let mut trial = directions.clone();
        trial.push(momentum.clone());
        let next_rank = rank(&trial, external)?;
        if next_rank > current_rank {
            shifts.push((shifts.last().unwrap() + momentum).expand());
            directions.push(momentum.clone());
            masses.push(Atom::Zero);
            exponents.push(0);
            current_rank = next_rank;
        }
    }
    if directions.len() > 32 {
        return Err(invalid(
            "one-loop recurrence supports at most 32 external coordinate directions",
        ));
    }
    let backend_k = Atom::var(S.k);
    let backend_products = directions
        .iter()
        .enumerate()
        .map(|(index, _)| {
            scalar_product(
                &backend_k,
                &Atom::var(symbol!(format!("oneloopmaster::q{}", index + 1))),
            )
        })
        .collect::<Vec<_>>();
    let mut basis = Vec::new();
    let mut equations = Vec::new();
    for (index, direction) in directions.iter().enumerate() {
        let mut trial = basis.clone();
        trial.push(direction.clone());
        if rank(&trial, external)? > basis.len() {
            basis.push(direction.clone());
            equations.push(direction - &backend_products[index]);
        }
    }
    let images = if external.is_empty() {
        Vec::new()
    } else {
        let (matrix, rhs) =
            Atom::system_to_matrix::<u16, _, _>(&equations, external).map_err(invalid)?;
        matrix
            .solve(&rhs)
            .map_err(invalid)?
            .into_vec()
            .into_iter()
            .map(|value| value.to_expression().cancel())
            .collect::<Vec<_>>()
    };
    let coefficients = origin
        .coefficient_list::<i32>(external)
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let k_origin = external
        .iter()
        .zip(&images)
        .map(|(momentum, image)| coefficients.get(momentum).cloned().unwrap_or_default() * image)
        .sum::<Atom>();
    let mut rules = vec![
        scalar_product(&backend_k, &backend_k) - Atom::num(2) * k_origin
            + kin.scalar_product(&origin, &origin).map_err(invalid)?,
    ];
    for (momentum, image) in external.iter().zip(images) {
        rules.push(image - kin.scalar_product(&origin, momentum).map_err(invalid)?);
    }
    let mut transformed = Atom::Zero;
    for (key, coefficient) in numerator.coefficient_list::<i32>(family.scalar_products()) {
        let powers = monomial_powers(key.as_view(), family.scalar_products())?;
        let mut term = coefficient;
        for (rule, power) in rules.iter().zip(powers) {
            term *= rule.pow(power);
        }
        transformed += term;
    }
    let mut invariants = Vec::new();
    for (i, first) in shifts.iter().enumerate() {
        for second in &shifts[i + 1..] {
            let difference = first - second;
            invariants.push(
                kin.scalar_product(&difference, &difference)
                    .map_err(invalid)?,
            );
        }
    }
    // Preserve generated scalar products while shielding ordinary user scalars.
    let mut chart_products = backend_products;
    chart_products.push(scalar_product(&backend_k, &backend_k));
    let backend = RecurrenceInput {
        masses_squared: masses
            .iter()
            .map(|mass| hygiene.protect(mass, &[]))
            .collect(),
        invariants: invariants.iter().map(|s| hygiene.protect(s, &[])).collect(),
        powers: exponents,
        numerator: hygiene.protect(&transformed.expand(), &chart_products),
    };
    let prefactor = hygiene.protect(&prefactor, &[]);
    let mut result = reduce(&backend)?;
    for (coefficient, _) in &mut result.terms {
        *coefficient *= &prefactor;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MasterBasis, MasterIntegral, OneLoopMasters};
    use feynkit_kinematics::Kinematics;

    #[test]
    fn signed_power_limits_are_checked_before_expansion() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_reduction_test::D"));
        let momentum = Atom::var(symbol!("shared_reduction_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family =
            SharedFamily::new(vec![momentum], vec![], vec![squared - Atom::num(2)], &kin).unwrap();
        for power in [i32::MIN, -21, 33, i32::MAX] {
            assert!(reduce_family(&family, &[power], &Atom::num(1)).is_err());
        }
    }

    #[test]
    fn user_parameters_do_not_capture_recurrence_coordinates_or_dimension() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_reduction_test::D"));
        let momentum = Atom::var(symbol!("shared_reduction_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        for name in ["d", "xll", "xq1", "den1", "reg_delta"] {
            let parameter = Atom::var(symbol!(format!("oneloopmaster::{name}")));
            let family = SharedFamily::new(
                vec![momentum.clone()],
                vec![],
                vec![&squared - Atom::num(2)],
                &kin,
            )
            .unwrap();
            let reduced = reduce_family(&family, &[2], &parameter).unwrap();
            let expression = reduced
                .terms
                .into_iter()
                .map(|(coefficient, master)| coefficient * OneLoopMasters.symbol(&master))
                .sum::<Atom>();
            let expected = &parameter * (&dimension - Atom::num(2)) / Atom::num(4)
                * OneLoopMasters.symbol(&MasterIntegral::Tadpole { m_sq: Atom::num(2) });
            assert!(
                (expression - expected).cancel().is_zero(),
                "scalar parameter {name}"
            );

            // Restoration also traverses master arguments, including masses.
            let family = SharedFamily::new(
                vec![momentum.clone()],
                vec![],
                vec![&squared - &parameter],
                &kin,
            )
            .unwrap();
            let reduced = reduce_family(&family, &[1], &Atom::num(1)).unwrap();
            assert_eq!(
                reduced.terms,
                vec![(Atom::num(1), MasterIntegral::Tadpole { m_sq: parameter })]
            );
        }
    }
}
