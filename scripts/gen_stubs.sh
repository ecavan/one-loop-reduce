#!/usr/bin/env bash
# Regenerate python/symbolica/community/oneloopreduce/__init__.pyi.
#
# pyo3-stub-gen's `define_stub_info_gatherer!` hard-codes
# `$CARGO_MANIFEST_DIR/pyproject.toml` as the place it reads the module name and
# the Python source root from, and it needs a binary target to run in. Neither
# belongs in a crate that is only ever linked into symbolica-community, so this
# script scaffolds both, runs the generator, and takes them away again.
#
# Once the crate is wired into a symbolica-community checkout, that root's own
# `cargo run --bin stub_gen --features python_stubgen` regenerates this file
# along with every other community module's, and is the authoritative path.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$repo/crates/one-loop-reduce-python"

cleanup() {
    rm -f "$crate/pyproject.toml" "$crate/src/bin/stub_gen.rs"
    rmdir "$crate/src/bin" 2>/dev/null || true
    rm -f "$repo/python/symbolica/core.pyi"
}
trap cleanup EXIT

cat > "$crate/pyproject.toml" <<'TOML'
[project]
name = "symbolica"
[tool.maturin]
module-name = "symbolica.core"
python-source = "../../python"
TOML

mkdir -p "$crate/src/bin"
cat > "$crate/src/bin/stub_gen.rs" <<'RUST'
fn main() -> pyo3_stub_gen::Result<()> {
    oneloopreduce_python::stub_info()?.generate()?;
    Ok(())
}
RUST

cargo run --manifest-path "$repo/Cargo.toml" \
    -p one-loop-reduce-python --features python_stubgen --bin stub_gen

# The generator writes a flat `<module>.pyi`; the community tree keeps each
# module in its own package next to its `__init__.py`. It also re-emits
# symbolica's own `core.pyi` from the linked symbolica, which is not ours to
# ship -- `cleanup` drops it.
mkdir -p "$repo/python/symbolica/community/oneloopreduce"
mv "$repo/python/symbolica/community/oneloopreduce.pyi" \
   "$repo/python/symbolica/community/oneloopreduce/__init__.pyi"

echo "wrote python/symbolica/community/oneloopreduce/__init__.pyi"
