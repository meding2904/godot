def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "WaterBody3D",
        "WaterBuoyancy3D",
        "WaterBoat3D",
        "WaterWake3D",
        "WaterSpray3D",
    ]


def get_doc_path():
    return "doc_classes"
