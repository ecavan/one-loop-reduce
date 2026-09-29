use symbolica::atom::Atom;

use crate::error::OneLoopError;
use crate::masters::{MasterBasis, OneLoopMasters};
use crate::recurrence::RecurrenceInput;
use crate::reduce::reduce;

pub fn amplitude(family: &RecurrenceInput) -> Result<Atom, OneLoopError> {
    let basis = OneLoopMasters;
    Ok(reduce(family)?
        .terms
        .iter()
        .fold(Atom::Zero, |acc, (coeff, master)| {
            acc + coeff * basis.symbol(master)
        }))
}
