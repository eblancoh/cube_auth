# Cube Auth Backend: Rubik’s Cube Machine Learning Web Authentication

PoC for Web Authentication based on Machine Learning analysis of 3x3 Rubik's Cube solving sequences. This repo includes the code for Authentication Backend Machine Learning component of the system.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing

The software in this repo can be executed in Windows or Linux.

No special HW or SW prerequisites are required to execute all the routines that constitute the Backend.

The Python version used to develop and test this repo is **Python 3.6.7**

#### Installation of Anaconda Distribution

Anaconda Distribution was used to set up the Python environment used to created this component of  the Cueb Auth PoC. It can be deployed on Linux, Windows, and Mac OS X.

**For Linux distro**

Tested on [Ubuntu Desktop 16.04 LTS](http://releases.ubuntu.com/16.04/) and [Ubuntu 18.04.1 LTS](https://www.ubuntu.com/download/desktop)


Update the package lists for upgrades for packages that need upgrading, as well as new packages that have just come to the repositories.

```
$ sudo apt-get update
```

Fetch new versions of packages existing on the machine

```
$ sudo apt-get upgrade
```


Our next step will be to download and install Anaconda Python 3.4

```
$ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
$ bash Anaconda3-4.2.0-Linux-x86_64.sh
```

Anaconda will incorporate automatically the routes to .bashrc file if specified during installation process.
In case this didn't work, it can be manually added after installation procedure as indicated hereafter:

```
$ gedit ~/.bashrc
```
So the next line is added to the file

```
export PATH="/home/$USER/anaconda3/bin:$PATH"
```

**For Windows OS**

This software was tested in Windows 10.

Download the [installer](https://repo.continuum.io/archive/Anaconda3-4.2.0-Windows-x86_64.exe), double-click the .exe file and follow the instructions on the screen.

If you are unsure about any setting, accept the defaults. You can change them later.

When installation is finished, from the Start menu, open the Anaconda Prompt.

#### Building the virtual environment with Conda

Open a terminal and, with the provided _environment.yml_ file, run:

```
conda env create -f environment.yml
```

A new virtual environment called _cubeauth_ is created.


#### Activate the new environment

On Linux:

```
source activate cubeauth
```

On Windows: 

```
activate cubeauth
```


#### New environment's correct installation cross-check

```
conda list
```

To verify that the copy was made:
```
conda info --envs
```

#### Install remaining dependencies

Within the _cubeauth_ environment, run.
```
pip install -r requirements.txt
```

## Running the app

Once the environment has been correctly installed, activate the virtual enviroment and run:


```
python ~/cube_auth/app/app.py
```

The app will be running and listening, prompting the following message in the terminal:
```
Using TensorFlow backend.
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```


## Built With

* [Python](https://www.python.org/) - Programming Language
* [Flask](http://flask.pocoo.org/) - Microframework for Python Web development based on Werkzeug, Jinja 2 and good intentions.
* [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL.

## Contributing

TBD


## Authors

* **Enrique Blanco Henríquez** - *Telefónica Digital España - CDO* 

    [GitHub - eblancoh](https://github.com/eblancoh)


## License

This project is licensed as GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This software doesn't have a QA Process. This software is a Proof of Concept.

For more information please visit http://www.elevenpaths.com