# PCC-Photobooth

A project for Computer hardware design class at Kmitl Chumphon.
Using gphoto2 as a backed to talk to DSLR and Flask for webinterface 

## Using the project

First Create [Python virtual environment](https://virtualenv.pypa.io/en/latest/) and activate them

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then use [pip](https://pip.pypa.io/en/stable/) to install requirement.

```bash
pip3 install -r requirements.txt
```

To start a server run

```bash
python3 server.py
```

please unsure that port 8080 is not use by other application and camera connected

## Todo list

- [x] Init gphoto2
- [x] Create basic web server
- [x] Add basic overlay
- [ ] Make setting interface
- [ ] Improve user interface
- [ ] do sql for database
- [ ] make user loginable and history
- [ ] paying system

## License

[GPL-3.0](LICENSE.md)