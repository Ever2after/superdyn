"""Minimal native Vulkan probe. No engine, NumPy, display server or admin install."""

import ctypes as c, json


class App(c.Structure):
    _fields_ = [
        ("type", c.c_uint32),
        ("next", c.c_void_p),
        ("app", c.c_char_p),
        ("version", c.c_uint32),
        ("engine", c.c_char_p),
        ("engine_version", c.c_uint32),
        ("api", c.c_uint32),
    ]


class Create(c.Structure):
    _fields_ = [
        ("type", c.c_uint32),
        ("next", c.c_void_p),
        ("flags", c.c_uint32),
        ("app", c.POINTER(App)),
        ("layers_n", c.c_uint32),
        ("layers", c.c_void_p),
        ("extensions_n", c.c_uint32),
        ("extensions", c.c_void_p),
    ]


class Queue(c.Structure):
    _fields_ = [
        ("flags", c.c_uint32),
        ("count", c.c_uint32),
        ("timestamp_bits", c.c_uint32),
        ("granularity", c.c_uint32 * 3),
    ]


try:
    v = c.CDLL("libvulkan.so.1")
    app = App(0, None, b"SuperDynProbe", 1, b"", 0, (1 << 22) | (1 << 12))
    info = Create(1, None, 0, c.pointer(app), 0, None, 0, None)
    instance = c.c_void_p()
    v.vkCreateInstance.argtypes = [c.POINTER(Create), c.c_void_p, c.POINTER(c.c_void_p)]
    status = v.vkCreateInstance(c.byref(info), None, c.byref(instance))
    if status:
        raise RuntimeError("vkCreateInstance result " + str(status))
    v.vkEnumeratePhysicalDevices.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_uint32),
        c.c_void_p,
    ]
    count = c.c_uint32()
    v.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
    devices = (c.c_void_p * count.value)()
    v.vkEnumeratePhysicalDevices(instance, c.byref(count), devices)
    results = []
    v.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
    v.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_uint32),
        c.c_void_p,
    ]
    for i, device in enumerate(devices):
        properties = c.create_string_buffer(4096)
        v.vkGetPhysicalDeviceProperties(device, properties)
        name = properties.raw[20:276].split(b"\0")[0].decode(errors="replace")
        n = c.c_uint32()
        v.vkGetPhysicalDeviceQueueFamilyProperties(device, c.byref(n), None)
        queues = (Queue * n.value)()
        v.vkGetPhysicalDeviceQueueFamilyProperties(device, c.byref(n), queues)
        results.append(
            dict(
                index=i,
                name=name,
                graphics_queue=any(q.flags & 1 and q.count for q in queues),
                queue_flags=[q.flags for q in queues],
            )
        )
    v.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    v.vkDestroyInstance(instance, None)
    print(
        json.dumps(
            dict(
                status=(
                    "passed"
                    if any(d["graphics_queue"] for d in results)
                    else "no_graphics_queue"
                ),
                devices=results,
            )
        )
    )
except Exception as e:
    print(json.dumps(dict(status="unavailable", error=str(e))))
