import typer

app = typer.Typer()

@app.command()
def turbulence():
    print("HELLO from the other side")

@app.command()
def get_data():
    print("potatoes")
    


