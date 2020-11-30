import numpy as np

def sigmoid2 (x,r,s=20.):
    return 1. / (1 + np.exp(-s*(x-r)))

sigmoid = np.vectorize(sigmoid2, excluded = ["r"])

def make_lattice(L_x, L_y, L_z):
    if L_x == 1:
        xs = np.array([0.0])
    else:
        xs = np.linspace(0.,1.,L_x) - 0.5
    if L_y == 1:
        ys = np.array([0.0])
    else:
        ys = np.linspace(0.,1.,L_y) - 0.5
    if L_z == 1:
        zs = np.array([0.0])
    else:
        zs = np.linspace(0.,1.,L_z) - 0.5
    return np.meshgrid(xs, ys, zs, indexing = 'ij')

def make_skyrmion(xs, ys, zs, r, s = 7.):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi

    theta= np.pi * sigmoid(np.sqrt(xs * xs + ys * ys), r, s)
    theta[L_x//2, L_y//2, :] = 0.0
    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [0, 1, 2])
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_helical(xs, ys, zs, r, s =20.):
    L_x, L_y, L_z = xs.shape
    theta = np.transpose(np.broadcast_to( np.linspace(0,2*np.pi,L_x+1)[:-1], (L_z,L_y,L_x)),(2, 1, 0))

    theta= np.pi * sigmoid(np.sqrt(xs * xs), r, s)
    theta[L_x//2, L_y//2, :] = 0.0
    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = np.transpose(np.arctan2(xs, 0.0)[::-1], axes = [0, 1, 2])
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    theta[-1] = np.arccos(-w[:-1].sum())
    print("work or not:", theta[-1])
    print("work or not:", -w[:-1].sum())
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_helical2(xs, ys, zs, r, s =20.):
    L_x, L_y, L_z = xs.shape
    theta = np.transpose(np.broadcast_to( np.linspace(0,2*np.pi,L_x+1)[:-1], (L_z,L_y,L_x)),(2, 1, 0))

    theta= 2*(L_x-1)/(L_x)*np.pi *np.abs(xs)
    #theta[L_x//2, L_y//2, :] = 0.0
    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = -np.transpose(np.arctan2(xs, 0.0)[::-1], axes = [0, 1, 2])
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_atanhelical(xs, ys, zs, r, s =20.):
    L_x, L_y, L_z = xs.shape
    theta = np.transpose(np.broadcast_to( np.linspace(0,2*np.pi,L_x+1)[:-1], (L_z,L_y,L_x)),(2, 1, 0))
    print(xs)
    mid = - xs.flatten()[0:len(xs.flatten())//2].mean()
    theta= np.pi/2.-np.arctan(40*(np.abs(xs)-mid))
    #theta[L_x//2, L_y//2, :] = 0.0
    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = -np.transpose(np.arctan2(xs, 0.0)[::-1], axes = [0, 1, 2])
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_conical(xs, ys, zs, m = 0.4):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi
    theta[:, :, :] = np.arccos(m)

    theta = theta.flatten()

    phi = np.copy(theta.reshape(L_x, L_y, L_z)[:,:,:])
    #phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [1, 0, 2])
    phi = np.broadcast_to(np.linspace(0,2*np.pi,L_z+1)[:-1],(L_x, L_y, L_z))#
    phi = phi.flatten()
    u = np.sin(phi) * np.sin(theta)
    v = np.cos(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_chiralbubble(xs, ys, zs, r, s = 7.):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi
    rs = np.linspace(0.08 , r, L_z -1)
    layer = 0
    theta[:,:, layer]= np.pi * 1.0
    for ri in rs:
        layer += 1
        theta[:,:, layer]= np.pi * sigmoid(np.sqrt(xs[:,:,0] * xs[:,:,0] + ys[:,:,0] * ys[:,:,0]), ri, s)
        theta[L_x//2, L_y//2, layer] = 0.0

    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [0, 1, 2])
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_para(xs, ys, zs, m = 1.0):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi
    theta[:, :, :] = np.arccos(m)

    theta = theta.flatten()

    phi = np.copy(theta.reshape(L_x, L_y, L_z)[:,:,:])
    #phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [1, 0, 2])
    phi = 2*np.pi * np.random.rand(L_x, L_y, L_z) * 0.0#
    phi = phi.flatten()
    u = np.sin(phi) * np.sin(theta)
    v = np.cos(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_para2(xs, ys, zs, m=0.5):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi
    theta[:, :, :] = np.arccos(m)

    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [0, 1, 2])
    phi = 2*np.pi * np.random.rand(L_x, L_y, L_z)# * 0.0#
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def make_para3(xs, ys, zs, m=0.0):
    L_x, L_y, L_z = xs.shape
    theta = np.ones((L_x, L_y, L_z)) * np.pi
    theta[:, :, :] = np.pi * np.random.rand(L_x, L_y, L_z)

    theta = theta.flatten()

    phi = theta.reshape(L_x, L_y, L_z)[:,:,:]
    phi = np.transpose(np.arctan2(xs, ys)[::-1], axes = [0, 1, 2])
    phi = 2 * np.pi * np.random.rand(L_x, L_y, L_z)# * 0.0#
    phi = phi.flatten()
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    theta[-1] = np.arccos(-w[:-1].sum())
    print("work or not:", theta[-1])
    u = np.cos(phi) * np.sin(theta)
    v = np.sin(phi) * np.sin(theta)
    w = np.cos(theta)
    return u, v, w

def read_config():
    config = np.load("config.npy")
    Lx, Ly, Lz = config[0].shape
    u = config[0].flatten()
    v = config[1].flatten()
    w = config[2].flatten()
    return u, v, w, Lx, Ly, Lz


def make_configuration(config_type, xs, ys, zs, r = 0.2, s = 20., m = 1.0):

    if config_type == "skyrmion":
        return make_skyrmion(xs, ys, zs, r, s)
    if config_type == "helical":
        return make_helical(xs, ys, zs, r, s)
    if config_type == "helical2":
        return make_helical2(xs, ys, zs, m)
    if config_type == "atanhelical":
        return make_atanhelical(xs, ys, zs, m)
    if config_type == "conical":
        return make_conical(xs, ys, zs, m)
    if config_type == "para":
        return make_para(xs, ys, zs, m)
    if config_type == "para2":
        return make_para2(xs, ys, zs, m)
    if config_type == "para3":
        return make_para3(xs, ys, zs, m)
    if config_type == "chiralbubble":
        return make_chiralbubble(xs, ys, zs, r, s)
    if config_type == "read":
        return read_config(xs, ys, zs)