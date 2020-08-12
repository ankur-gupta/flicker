# Flicker Playground
This docker image provides an environment to run pyspark and 
[flicker](https://flicker.perfectlyrandom.org/) 
without having to setup anything. This docker image is based on 
[pyspark-playground](https://github.com/ankur-gupta/pyspark-playground)
but focuses on a single container (instead of a cluster in 
`pyspark-playground`). 

Similar to 
[pyspark-playground](https://github.com/ankur-gupta/pyspark-playground),
`flicker-playground` should not be used in production. Please see 
[pyspark-playground](https://github.com/ankur-gupta/pyspark-playground)
for details on design of this docker image. 

## Features
* Ubuntu 20.04 LTS (Focal Fossa)
* tini
* Python 3.8 and `pip`
* Jupyter & IPython
* Basic python packages (`numpy`, `pandas`, `matplotlib`, `scikit-learn`)
* Java 8
* `pyspark` (latest)
* `flicker` (latest)

### User
This image creates a `sudo`-privileged user for you. You should be able to
do everything (including installing packages using `apt-get`) as this user
without having to become `root`.

| Key      | Value        |
|----------|--------------|
| Username | `neo`        |
| Password | `agentsmith` |

### Jupyter notebook
By default, running a container from this image would run a jupyter notebook
at port `8888`. The port `8888` is not exposed in the `Dockerfile` you can
expose it and bind it to a port on the host machine via command line.

## Getting the image
### From the internet
If you just want to use the docker image, you don't need build it yourself. 
The image may be pulled from 
[DockerHub](https://hub.docker.com/r/ankurio/flicker-playground).
```bash
# From DockerHub (https://hub.docker.com/r/ankurio/flicker-playground) 
docker pull ankurio/flicker-playground
```

### Build it yourself
1. **Install or update docker**. Docker from the
[official website](https://docs.docker.com/get-docker/) works well. Please
update your docker because because we use some of the newer features of
Docker Compose in this repository which may not be available with older
versions.

2. **Clone the repository**
    ```bash
    git clone git@github.com:ankur-gupta/flicker.git
    ```
3. **Build the image**
    ```bash
   cd $REPO_ROOT/docker
   docker build . -t flicker-playground
    ```
   Building the image will take a long time for the first time but repeated 
   builds (after minor edits to `Dockerfile`) should be quick because 
   every layer gets cached.

   Check that the docker image was built successfully
   ```bash
   docker images pyspark-playground
   # REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
   # flicker-playground   latest              8fa20519926d        15 minutes ago      1.39GB
   ```

4. **Test the image**
   ```bash
   # On your host machine
   docker run -it -p 8888:8888 flicker-playground
   # ...
   # http://127.0.0.1:8888/?token=s0m3a1phanum3rict0k3n
   ```
   Use your browser to go to the address printed in terminal.

   Exit the container by pressing `Control+C` in the
   terminal. Exiting is important because the above command binds host
   machine's port `8888` and as long as this container is running you won't be
   able to bind anything else on the same port.

   You can also get to a shell prompt instead of running a Jupyter
   notebook. You can run `ipython` or scripts in the terminal.
   ```bash
   # On your host machine
   docker run -it flicker-playground /bin/bash
   # To run a command as administrator (user "root"), use "sudo <command>".
   # See "man sudo_root" for details.
   # neo@1ebdf087682d:~$  
   ```
   Press `Control+D` to exit.
   
## Using the image
If you built the image, the image name is whatever you supplied; we used the
image name `flicker-playground` in the instructions above. If you pulled the
image from DockerHub, then the image name should be 
`ankurio/flicker-playground`. Use either image name in the following 
instructions; we use `ankurio/flicker-playground`.

### Run Jupyter Notebook with a data mount
```bash
# From $REPO_ROOT/docker in the host machine
docker run -it \
-p 8888:8888 \
-p 8080:8080 \
-p 4040:4040 \
--volume $PWD/notebooks:/home/neo/notebooks \
ankurio/flicker-playground
```

### Run IPython with a data mount
```bash
# From $REPO_ROOT/docker in the host machine
docker run -it --volume $PWD/notebooks:/home/neo/notebooks ankurio/flicker-playground /bin/bash
# Run ipython on the shell prompt
# neo@95657e2ed2c6:~$ ipython 
``` 

## Known Issues
### Why is there no `https://` ?
Both Jupyter notebook and Spark serve web pages. These web pages are served
on `http://` instead of `https://`, by default. For Jupyter, this can be
fixed as shown in
[pyspark-notebook](https://github.com/jupyter/docker-stacks/blob/master/base-notebook/jupyter_notebook_config.py#L18)
but this hasn't been implemented yet. For Spark web UIs, this is more
difficult as mentioned
[here](https://stackoverflow.com/questions/44936756/how-to-configure-spark-standalones-web-ui-for-https).
Spark 3.0 is here and this issue ourselves. See `$REPO_ROOT/index.html` for a handy list
of all possible URLs.
