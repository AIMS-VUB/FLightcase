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
from FLightcase.utils.template_creation import fill_or_copy, copy_template


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


@cli.command()
@click.option('--workspace_path', required=True, type=str, help='Path to workspace')
@click.option('--who', required=True, type=str, help='"server" or "client"')
def prepare_workspace(workspace_path, who):
    """Prepare workspace"""

    # Create workspace if it does not yet exist
    if not os.path.exists(workspace_path):
        print(f'\nCreating workspace directory: {workspace_path}...\n')
        os.makedirs(workspace_path)

    if who == 'server':
        fill_or_copy(workspace_path, 'FL_plan.json')
        fill_or_copy(workspace_path, 'server_node_settings.json')
        copy_template(workspace_path, 'architecture.py')
    elif who == 'client':
        fill_or_copy(workspace_path, 'client_node_settings.json')


if __name__ == '__main__':
    cli()
