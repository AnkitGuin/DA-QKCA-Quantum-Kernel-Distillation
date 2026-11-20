from daqkca.utils.config import config_hash

def test_config_hash_independent_of_dict_order():
    assert config_hash({'a':1,'b':2})==config_hash({'b':2,'a':1})
