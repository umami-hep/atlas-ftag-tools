from ftag.cuts import Cut, Cuts
from ftag.region import Region


def test_region_str():
    cut1 = Cut("var1", ">", 5)
    cut2 = Cut("var2", "<", 3)
    cuts = Cuts((cut1, cut2))
    region = Region("TestRegion", cuts)
    assert str(region) == "TestRegion"


def test_region_comparison():
    cut1 = Cut("var1", ">", 5)
    cut2 = Cut("var2", "<", 3)
    cuts1 = Cuts((cut1, cut2))
    region1 = Region("TestRegion1", cuts1)

    cut3 = Cut("var1", "<", 2)
    cut4 = Cut("var2", ">", 4)
    cuts2 = Cuts((cut3, cut4))
    region2 = Region("TestRegion2", cuts2)

    assert region1 < region2


def test_region_equality():
    cut1 = Cut("var1", ">", 5)
    cut2 = Cut("var2", "<", 3)
    cuts1 = Cuts((cut1, cut2))
    region1 = Region("TestRegion", cuts1)

    cut3 = Cut("var1", ">", 5)
    cut4 = Cut("var2", "<", 3)
    cuts2 = Cuts((cut3, cut4))
    region2 = Region("TestRegion", cuts2)

    assert region1 == region2
