"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import git, os, sys
from git import RemoteProgress

def git_updater(directory):
    try:
        print("Checking for updates...")
        restart_needed = False
        repo = git.Repo(directory)
        for fetch_info in repo.remotes.origin.fetch(progress=RemoteProgress()):
            print(f"Updated {fetch_info.ref} to {fetch_info.commit}")

        local_head = repo.heads[0].commit
        remote_head = repo.remotes.origin.refs[0].commit
        print(f"Local commit: {local_head}")
        print(f"Last remote commit: {remote_head}")

        for fetch_info in repo.remotes.origin.pull(repo.heads[0], progress=RemoteProgress()):
            print(f"Updated {fetch_info.ref} to {fetch_info.commit}")
            if fetch_info.commit != local_head:
                restart_needed = True
                print(f"Local head moved to {local_head}. The program will be restarted.")

        print("Checking for updates in submodules...")
        for submodule in repo.submodules:
            for fetch_info in submodule.update(init = True, recursive = True):
                print(f"Updated {fetch_info.ref} to {fetch_info.commit}")
                if fetch_info.commit != local_head:
                    restart_needed = True
                    print(f"Local head moved to {local_head}. The program will be restarted.")

        return restart_needed
    except git.exc.GitCommandError as e:
        print(e)
        return False