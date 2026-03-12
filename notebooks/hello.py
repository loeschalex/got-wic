import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md("# Hello, marimo!")
    return (mo,)


@app.cell
def _(mo):
    name = mo.ui.text(placeholder="your name", label="What's your name?")
    name
    return (name,)


@app.cell
def _(mo, name):
    mo.md(f"Hello, **{name.value or 'world'}**! 👋") if name.value else mo.md(
        "Type your name above!"
    )
    return


if __name__ == "__main__":
    app.run()
