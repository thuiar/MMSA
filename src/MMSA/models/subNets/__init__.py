from .BertTextEncoder import BertTextEncoder
from .FeatureNets import SubNet, TextSubNet
from .AlignNets import AlignSubNet


TRANSFORMERS_MAP = {
    'bert': BertTextEncoder,
}