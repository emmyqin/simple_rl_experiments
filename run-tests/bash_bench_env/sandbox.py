from pathlib import Path
from uuid import uuid4
from os.path import dirname, abspath
import subprocess
from beartype import beartype
from dataclasses import dataclass


@dataclass
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


@beartype
class DockerSandbox:
    container_name: str

    def __init__(self):
        self.container_name = ""
        self.container_name = f"bash-sandbox-instance-{uuid4()}"
        print(f"CREATING SANDBOX {self.container_name}")
        self.build_image()
        self.start_container()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def build_image(self) -> None:
        sandbox_path = Path(dirname(abspath(__file__)) + "/sandbox")
        if not sandbox_path.is_dir():
            raise FileNotFoundError(f"Sandbox directory '{sandbox_path}' not found.")
        
        result = subprocess.run(
            ["docker", "build", "-t", "bash-sandbox", str(sandbox_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Error building image: {result.stderr}")

    def start_container(self) -> None:
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "--tty",
                "bash-sandbox",
                "/bin/sh",
                "-c",
                "while true; do sleep 1; done"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Error starting container: {result.stderr}")

    def run_command(self, command: str, timeout_seconds: int = 30) -> CompletedProcess:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_name,
                    "/bin/bash",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            return CompletedProcess(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            return CompletedProcess(
                returncode=1,
                stdout="",
                stderr="Timed out."
            )

    def cleanup(self) -> None:
        docker_stop_response = subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True,
            text=True
        )

        docker_remove_response = subprocess.run(
            ["docker", "rm", self.container_name],
            capture_output=True,
            text=True
        )

        process = subprocess.Popen(
            f"docker stop {self.container_name}; docker rm {self.container_name}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
