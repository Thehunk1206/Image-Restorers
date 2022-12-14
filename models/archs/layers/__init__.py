try:
    from layers.edge_padding2D import EdgePadding2D
    from layers.tlc_avgpool import TlcAvgPool2D
    from layers.pixel_shuffle import PixelShuffle
    from layers.simplified_channel_attention import SimplifiedChannelAttention
    from layers.naf_block import NAFBlock
    from layers.simple_gate import SimpleGate
except:
    from archs.layers.edge_padding2D import EdgePadding2D
    from archs.layers.tlc_avgpool import TlcAvgPool2D
    from archs.layers.pixel_shuffle import PixelShuffle
    from archs.layers.simplified_channel_attention import SimplifiedChannelAttention
    from archs.layers.naf_block import NAFBlock
    from archs.layers.simple_gate import SimpleGate