import typer

# Root application for this interface
lite_app = typer.Typer(help="Lite run of rojak for lower memory usage")

# Turbulence Functionality
turbulence_app = typer.Typer(help="Computations related to turbulence")

# Add applications related to lite app here
lite_app.add_typer(turbulence_app, name="turbulence")
