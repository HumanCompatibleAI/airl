import os

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_xml(xml_name):
  return os.path.join(ENV_ASSET_DIR, xml_name)
