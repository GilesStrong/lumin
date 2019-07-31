import pkg_resources


def check_pdpbox():
    try:
        ver = pkg_resources.get_distribution("pdpbox").version
        assert ver >= '0.2.0+13.g73c6966', f'''You have version {ver} of pdpbox. Use of this function requires pdpbox>=0.2.0+13.g73c6966, which is not currently
                                                available from pypi. Please install from source via:\n 
                                               `git clone https://github.com/SauceCat/PDPbox.git && cd PDPbox && pip install -e .`'''
    except pkg_resources.DistributionNotFound:
        print('''Use of this function requires pdpbox>=0.2.0+13.g73c6966, which is not currently available from pypi. Please install from source via:\n 
                `git clone https://github.com/SauceCat/PDPbox.git && cd PDPbox && pip install -e .`''')
                