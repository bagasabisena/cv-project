__author__ = 'bagas'

import hashlib

sha256_actual = '8548658ef00f2ac4c384fbfff9d3ae225b4b9e0c2aa45e79a97420381c0f84c9'

with open('/Volumes/My Passport/facescrub/actor_3.jpg', 'rb') as f:
    data = f.read()
    sha256_ret = hashlib.sha256(data).hexdigest()
    print sha256_ret
    print sha256_actual

    if sha256_ret == sha256_actual:
        print 'real'
    else:
        print 'fake'