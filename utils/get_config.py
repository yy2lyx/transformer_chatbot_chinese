from configparser import ConfigParser
from ast import literal_eval


def get_conf(conf_path = 'conf.ini'):
    config = ConfigParser()
    config.read(conf_path,encoding='utf8')
    conf_dict = dict(config.items('CONF'))
    for k,v in conf_dict.items():
        try:
            v_new = literal_eval(v)
            conf_dict[k] = v_new
        except:
            pass
    return conf_dict
