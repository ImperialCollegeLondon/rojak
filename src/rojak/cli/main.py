import typer

from rojak.cli import data_interface

app = typer.Typer()
app.add_typer(data_interface.data_app, name="data")

@app.command()
def turbulence():
    print("HELLO from the other side")

@app.command()
def get_data():
    print("potatoes")
    


