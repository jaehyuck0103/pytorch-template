from box import Box

settings = Box()


def merge_config_from_toml(tomlpath):
    settings.merge_update(Box.from_toml(filename=tomlpath))
