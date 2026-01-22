"""
Command Line Interface (CLI). Easy to use + accessible remotely

Sources:
- Nice instructional video: https://www.youtube.com/watch?v=FWacanslfFM
- https://dojofive.com/blog/3-steps-to-make-a-professional-cli-tool-using-pythons-click/
"""


import os
import click
from FLightcase.server import server
from FLightcase.client import client


@click.group()
def cli():
    pass


@cli.command()
@click.option('--settings_path', type=str, help='Path to the server settings JSON')
def run_server(settings_path):
    """Run the server"""
    server(settings_path)


@cli.command()
@click.option('--settings_path', type=str, help='Path to the client settings JSON')
def run_client(settings_path):
    """Run the client"""
    client(settings_path)


if __name__ == '__main__':
    cli()
