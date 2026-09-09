# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "marimo",
#     "psyneulink",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    mo.md(
        """
        # Getting Started with PsyNeuLink

        This notebook verifies your local PsyNeuLink installation and
        demonstrates basic usage.

        **Setup:** Run `make install-tutorial` from the repo root, then
        `make marimo` to launch this notebook.
        """
    )
    return (mo,)


@app.cell
def _():
    # Suppress harmless FutureWarnings from graph-scheduler on Python 3.13+
    import warnings
    warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor")

    import psyneulink as pnl
    print(f"PsyNeuLink version: {pnl.__version__}")
    return (pnl,)


@app.cell
def _(mo):
    mo.md("""
    ## Create a simple Transfer Mechanism

    A `TransferMechanism` is one of the most basic components in PsyNeuLink.
    It takes an input, applies a function, and produces an output.
    """)
    return


@app.cell
def _(pnl):
    # Create a simple mechanism with a logistic function
    my_mech = pnl.TransferMechanism(
        name='my_mechanism',
        function=pnl.Logistic(gain=1.0, bias=0)
    )
    print(my_mech)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Build a simple Composition

    A `Composition` connects mechanisms together into a runnable model.
    """)
    return


@app.cell
def _(pnl):
    # Create two mechanisms
    input_mech = pnl.TransferMechanism(name='input', function=pnl.Linear)
    output_mech = pnl.TransferMechanism(name='output', function=pnl.Logistic)

    # Build a composition and connect them
    comp = pnl.Composition(name='simple_composition')
    comp.add_linear_processing_pathway([input_mech, output_mech])

    # Run with some input
    result = comp.run(inputs={input_mech: [1.0]})
    print(f"Output: {result}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    - See `tutorial/PsyNeuLink Tutorial.ipynb` for a comprehensive walkthrough
    - Browse the [PsyNeuLink documentation](https://princetonuniversity.github.io/PsyNeuLink/)
    - Add your own notebooks to this `notebooks/` directory
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
