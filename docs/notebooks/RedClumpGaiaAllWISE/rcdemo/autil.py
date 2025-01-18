
exec(open("./rcdemo/packages_to_import.py").read()) 



def xyz2lbr(x,y,z):
#    dtor=np.pi/180.0
    rc2=x*x+y*y
    return [np.degrees(np.arctan2(y,x)),np.degrees(np.arctan(z/np.sqrt(rc2))),np.sqrt(rc2+z*z)]




def dist2dmod(d):
    """
    Arguments:
        d (float)-  distance in kpc
    """
    return 5.0*np.log10(100.0*d)

def dmod2dist(dmod):
    """
    Arguments:
        dmod (float)-  distance modulus
    """
    return np.power(10.0,dmod/5.0-2.0)

