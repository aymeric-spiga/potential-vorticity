#! /usr/bin/env python

####################################################
# python code for calculating Ertel PV on isentropes
# from a netCDF output from a GCM
#
# the core of the calculations are adapted from
# the initial fork of M. Barlow's PV script
# https://zenodo.org/record/1069032
#
# Python-like interpolations are added
# Links to planetary constants through "planets" are added
# 
# Aymeric Spiga and Alexandre Boissinot
####################################################
import numpy as np
import netCDF4 as nc
import planets

####################################################
### NetCDF file that contains ps, u, v and T
ncfile = "Xhistins_270_red.nc"
ncfile = "Xhistins_172_red.nc"
ncfile = "diagfi210_240.nc"
### txt file that contains ap and bp coefficients
txtfile = "apbp.txt"
### planetary constants
#myp = planets.Saturn
#myp = planets.Planet() ; myp.ini("Saturn_dynamico",whereset="./")
myp = planets.Mars
### pressure grid
#p_upper,p_lower,nlev = 1e2,3.5e5,50 # whole atm
p_upper,p_lower,nlev = 1e-1,1e3,50 # whole atm
targetp1d = np.logspace(np.log10(p_lower),np.log10(p_upper),nlev)
### reference pressure
p0 = targetp1d[0]+1.
### potential temperature grid
profile_1d = None
profile_1d = np.array([220,240,260,280,300,320])
profile_1d = np.linspace(220,520,20)
print profile_1d
### output file name
output = "PVnew_"+ncfile
####################################################

# =====================
#  --- FUNCTIONS ---
# =====================

####################################################
def getp_fromapbp(txtfile, ps):
   #ap,bp = np.loadtxt(txtfile,unpack=True)
   file = nc.Dataset(ncfile,'r')
   ap  = file.variables['ap'][:]
   bp  = file.variables['bp'][:]
   nz = len(ap)
   aps = 0.5*(ap[0:nz-1]+ap[1:nz])
   bps = 0.5*(bp[0:nz-1]+bp[1:nz])
   nz = len(aps)
   #print "... ps"
   nt,ny,nx = ps.shape
   p = np.zeros((nt,nz,ny,nx))
   #print "... compute p"
   for tt in range(nt):
      for kk in range(nz):
         p[tt,kk,:,:] = aps[kk]+bps[kk]*ps[tt,:,:]
   return p
####################################################
def interpolate4(targetp1d,sourcep3d,fieldsource3d,spline=False,log=False):
  if spline:
    from scipy import interpolate
  nt,nz,nlat,nlon = fieldsource3d.shape
  if log:
    coordsource3d = -np.log(sourcep3d) # interpolate in logp
    coordtarget1d = -np.log(targetp1d) # interpolate in logp
  else:
    coordsource3d = sourcep3d
    coordtarget1d = targetp1d
  nzt = coordtarget1d.size
  fieldtarget3d = np.zeros((nt,nzt,nlat,nlon))
  for mmm in range(nlon):
   for nnn in range(nlat):
    for ttt in range(nt):
     xs = coordsource3d[ttt,:,nnn,mmm]
     ys = fieldsource3d[ttt,:,nnn,mmm]
     if not spline:
       fieldtarget3d[ttt,:,nnn,mmm] = np.interp(coordtarget1d,xs,ys,left=np.nan,right=np.nan)
     else:
       tck = interpolate.splrep(xs, ys, s=0)
       fieldtarget3d[ttt,:,nnn,mmm] = interpolate.splev(coordtarget1d, tck, der=0)
  return fieldtarget3d
####################################################

# =====================
#  --- MAIN CODE ---
# =====================


#constants
re=myp.a
g=myp.g
kap=myp.R/myp.cp
omega=myp.omega
pi=np.pi

# open dataset, retreive variables, close dataset
print '... open dataset and get variables'
file = nc.Dataset(ncfile,'r')
lat  = file.variables['latitude'][:]
lon  = file.variables['longitude'][:]
tps  = file.variables['Time'][:]
tdim = len(tps)
xdim = len(lon)
ydim = len(lat)
ps = file.variables['ps'][:,:,:]
p = getp_fromapbp(txtfile, ps)
t = file.variables['temp'][:,:,:,:]
u = file.variables['u'][:,:,:,:]
v = file.variables['v'][:,:,:,:]
dustq = file.variables['dustq'][:,:,:,:]
h2o_ice = file.variables['h2o_ice'][:,:,:,:]
file.close()

# some prep work for derivatives
print '... prep work'
xlon,ylat=np.meshgrid(lon,lat)
dlony,dlonx=np.gradient(xlon)
dlaty,dlatx=np.gradient(ylat)
dx=re*np.cos(ylat*pi/180.)*dlonx*pi/180.
dy=re*dlaty*pi/180.

# define potential temperature and Coriolis parameter
theta=myp.tpot(t,p,p0=p0)
f=myp.fcoriolis(lat=ylat)

# define potential temperature grid (before interpolating)
if profile_1d is None:
  theta_min = 10.*np.floor(np.min(theta) / 10.)
  theta_max = 10.*np.ceil(np.max(theta) / 10.)
  theta_step = 20.
  profile_1d = np.arange(theta_min, theta_max+theta_step, theta_step)
  print "no tpot target profile, I calculated it: ", profile_1d
zdim = len(profile_1d)

# interpolate on pressure grid
print '... interpolating on pressure grid'
u = interpolate4(targetp1d, p, u, log=True)
v = interpolate4(targetp1d, p, v, log=True)
theta = interpolate4(targetp1d, p, theta, log=True)
dustq = interpolate4(targetp1d, p, dustq, log=True)
h2o_ice = interpolate4(targetp1d, p, h2o_ice, log=True)
p = interpolate4(targetp1d, p, p, log=True)


# calculate derivatives
# (np.gradient can handle 1D uneven spacing,
# so build that in for p, but do dx and dy 
# external to the function since they are 2D)
print '... calculate derivatives'
ninc=1 # account for 4D fields
lev = targetp1d
ddp_theta=np.gradient(theta,lev,axis=0+ninc)
ddx_theta=np.gradient(theta,axis=2+ninc)/dx
ddy_theta=np.gradient(theta,axis=1+ninc)/dy
ddp_u=np.gradient(u,lev,axis=0+ninc)
ddp_v=np.gradient(v,lev,axis=0+ninc)
ddx_v=np.gradient(v,axis=2+ninc)/dx
ddy_ucos=np.gradient(u*np.cos(ylat*pi/180.),axis=1+ninc)/dy

# calculate contributions to PV and PV
print '... calculate PV'
relvort=ddx_v-(1/np.cos(ylat*pi/180.))*ddy_ucos
plavort=f
pv_zero=g*plavort*(-ddp_theta)
pv_one=g*relvort*(-ddp_theta)
pv_two=g*(ddp_v*ddx_theta-ddp_u*ddy_theta)
pv=pv_zero+pv_one+pv_two
pvr=pv_one+pv_two

# interpolate on isentropes
print '... interpolating on tpot grid'
pv = interpolate4(profile_1d,theta,pv,log=False)
pvr = interpolate4(profile_1d,theta,pvr,log=False)
p_t = interpolate4(profile_1d,theta,p,log=False)
u = interpolate4(profile_1d,theta,u,log=False)
v = interpolate4(profile_1d,theta,v,log=False)
dustq = interpolate4(profile_1d,theta,dustq,log=False)
h2o_ice = interpolate4(profile_1d,theta,h2o_ice,log=False)

# =====================
#  --- OUTPUT FILE ---
# =====================

print '... writing'

if('.nc' in output):
   datas = nc.Dataset(output      ,'w', format='NETCDF4')
else:
   datas = nc.Dataset(output+'.nc','w', format='NETCDF4')

datas.createDimension('longitude', xdim)
datas.createDimension('latitude', ydim)
datas.createDimension('tpot', zdim)
datas.createDimension('time', tdim)

longitude     = datas.createVariable('longitude', 'f4', 'longitude')
latitude      = datas.createVariable('latitude', 'f4', 'latitude')
tpot          = datas.createVariable('tpot', 'f4', 'tpot')
time          = datas.createVariable('time', 'f4', 'time')
pot_vorticity = datas.createVariable('PV', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
relative_pv   = datas.createVariable('relative_PV', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
pressure      = datas.createVariable('p', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
zonwind       = datas.createVariable('u', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
merwind       = datas.createVariable('v', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
qdust         = datas.createVariable('dustq', 'f4', ('time', 'tpot', 'latitude', 'longitude'))
qice         = datas.createVariable('h2o_ice', 'f4', ('time', 'tpot', 'latitude', 'longitude'))

longitude[:]     = lon
latitude[:]      = lat
tpot[:]          = profile_1d
time[:]          = tps
pot_vorticity[:] = pv
relative_pv[:]   = pvr
pressure[:]      = p_t
zonwind[:]       = u             
merwind[:]       = v 
qdust[:]         = dustq
qice[:]          = h2o_ice

longitude.units     = 'degrees east'
latitude.units      = 'degrees north'
time.units          = 's'
tpot.units          = 'K'
pot_vorticity.units = 'kg m$^2$ s$^{-1}$ K$^{-1}$'
relative_pv.units   = 'kg m$^2$ s$^{-1}$ K$^{-1}$'
pressure.units      = 'Pa'
zonwind.units       = 'm s$^{-1}$'
merwind.units       = 'm s$^{-1}$'
qdust.units         = 'kg/kg'
qice.units          = 'kg/kg'

datas.close()
