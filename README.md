# Cube Auth: 3x3x3 Cube Puzzle Authentication

PoC for Web Authentication based on Machine Learning analysis of 3x3 Rubik's Cube solving sequences. This repo includes the code for Authentication Backend Machine Learning component of the system.

Web Service Authentication Concept Test Service based on Machine Learning analysis of the movements of a 3x3x3 cube making use of the resolution sequences of cubes that can send both its rotation sequences and positioning through the BLE protocol is proposed. In order to make this possible, both a hardware development of the authentication device and software implementation of the platform, interface and authentication engine in the web service have been performed. In this work, a 3x3x3 cube has been designed and manufactured by ElevenPaths, called Cube11Paths. This device is capable of transmitting via BLE channel not only the sequences of turns, but also positioning sequences, which differentiates it from the rest of today’s commercial puzzles. To allow authentication in a Web Service using a 3x3x3 cube, a machine learning engine dedicated to binary classification is proposed through Logistic Regression, Support Vector Machine and Random Forest Classifier algorithms using the most representative characteristics of the resolutions of each user. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing

The software in this repo can be executed in Windows or Linux.

No special HW or SW prerequisites are required to execute all the routines that constitute the Backend.

The Python version used to develop and test this repo is **Python 3.6.7**

#### Installation of Anaconda Distribution

Anaconda Distribution was used to set up the Python environment used to created this component of  the Cube Auth PoC. It can be deployed on Linux, Windows, and Mac OS X.

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

Once the environment has been correctly installed, activate the virtual enviroment and for start the consumer that launches the training routine:


```
python ~/cube_auth/ml_engine/train.all.py
```





The app will be running and listening, prompting the following message in the terminal:
```
Using TensorFlow backend.
 [*] Waiting for messages. To exit press CTRL+C
```

Very similar when the testing sequence is sent to the machine learning engine:
```
python ~/cube_auth/ml_engine/receive.test.py
```

The app will be running and listening, prompting the following message in the terminal:
```
Using TensorFlow backend.
 [*] Waiting for messages. To exit press CTRL+C
```


## Built With

* [Python](https://www.python.org/) - Programming Language
* [Pika](https://pika.readthedocs.io/en/stable/) - Pika is a pure-Python implementation of the AMQP 0-9-1 protocol that tries to stay fairly independent of the underlying network support library. If you have not developed with Pika or RabbitMQ before, the Introduction to Pika documentation is a good place to get started.

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