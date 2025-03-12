FROM ubuntu:latest

# Update package list and install bash (if not already installed)
RUN apt-get update && \
    apt-get install -y bash && \
    apt-get install -y sudo software-properties-common

RUN sudo apt-get install -y python3-full python3-pip python-is-python3 python3-pytest

# Create a non-root user
RUN useradd -m sandboxuser

# Allow sandboxuser to run apt-get without password
RUN echo "sandboxuser ALL=(root) NOPASSWD: /usr/bin/apt-get" > /etc/sudoers.d/sandboxuser && \
    chmod 0440 /etc/sudoers.d/sandboxuser

# Switch to the non-root user
USER sandboxuser

# Set the working directory
WORKDIR /home/sandboxuser

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Start a shell by default
CMD ["/bin/bash"]
