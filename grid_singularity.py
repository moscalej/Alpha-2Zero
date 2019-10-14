from d3a.models.appliance.switchable import SwitchableAppliance
from d3a.models.area import Area
from d3a.models.strategy.storage import StorageStrategy
from d3a.models.strategy.load_hours import CellTowerLoadHoursStrategy, LoadHoursStrategy
from d3a.models.appliance.pv import PVAppliance
from d3a.models.strategy.pv import PVStrategy

def get_community_setup(config, comunity_name, consumers, providers, storage):
    pass


com1 = dict(
    config={},
    name='com1',
    consumers=[],
    providers=[],
    storage=[],

)

com2 = dict(
    config={},
    name='com2',
    consumers=[],
    providers=[],
    storage=[],

)


def get_setup(config):
    area = Area(
        'Grid',
        [
            get_community_setup(**com1),
            get_community_setup(**com2),
        ],
        config=config
    )
    return area
