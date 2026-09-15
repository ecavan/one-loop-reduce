use symbolica::atom::Atom;

use crate::error::OneLoopError;
use crate::family::IntegralFamily;
use crate::masters::{MasterBasis, OneLoopMasters};
use crate::reduce::reduce;

pub fn amplitude(family: &IntegralFamily) -> Result<Atom, OneLoopError> {
    let basis = OneLoopMasters;
    Ok(reduce(family)?
        .terms
        .iter()
        .fold(Atom::Zero, |acc, (coeff, master)| {
            acc + coeff * basis.symbol(master)
        }))
}
