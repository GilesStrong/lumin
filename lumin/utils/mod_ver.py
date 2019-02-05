import pkg_resources


def check_rfpimp():
    assert pkg_resources.get_distribution("rfpimp").version > '1.3', '''Use of this function requires rfpimp>1.3, which is not currently available from pypi. Please install from source via:\n 
                                                                          `git clone https://github.com/parrt/random-forest-importances.git && cd random-forest-importances/src && pip install .`'''


def check_pdpbox():
    assert pkg_resources.get_distribution("pdpbox").version > '2.0', '''Use of this function requires pdpbox>2.0, which is not currently available from pypi. Please install from source via:\n 
                                                                        `git clone https://github.com/SauceCat/PDPbox.git && cd PDPbox && pip install .`'''
