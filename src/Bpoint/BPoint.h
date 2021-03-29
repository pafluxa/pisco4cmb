#ifndef INCBPointh
#define INCBPointh

#include "jpl_eph.h"
#include <sofa.h>

#define UNIX_JD2000 946728000.0
#define TAIMGPS 19.0

enum ClockType {
  CLK_UTC,
  CLK_UT1,
  CLK_TAI,
  CLK_GPS
};

struct AstroTime {
  double jd_tt[2];
  double jd_ut1[2];
  // TAI-UTC
  double dtai;
  // UT1-UTC
  double dut1;
  ClockType clock;
};

typedef struct AstroTime AstroTime_t;


class BPoint
{

/*     astrom iauASTROM*   star-independent astrometry parameters:
**      pmt    double       PM time interval (SSB, Julian years) ***unused***
**      eb     double[3]    SSB to observer (vector, au)
**      eh     double[3]    Sun to observer (unit vector)        ***optional***
**      em     double       distance from Sun to observer (au)   ***optional***
**      v      double[3]    barycentric observer velocity (vector, c)
**      bm1    double       sqrt(1-|v|^2): reciprocal of Lorenz factor
**      bpn    double[3][3] bias-precession-nutation matrix
**      along  double       longitude + s' (radians)
**      xpl    double       polar motion xp wrt local meridian (radians)  ***optional***
**      ypl    double       polar motion yp wrt local meridian (radians)  ***optional***
**      sphi   double       sine of geodetic latitude
**      cphi   double       cosine of geodetic latitude
**      diurab double       magnitude of diurnal aberration vector  ***unused***
**      eral   double       "local" Earth rotation angle (radians)
**      refa   double       refraction constant A (radians)  ***unused***
**      refb   double       refraction constant B (radians)  ***unused***
*/

  private:
    char * jpl_ephem_path;
    double site_longitude;
    double site_latitude;
    double site_height;
    double tlight;
    double xpole;
    double ypole;
    double cip_x;
    double cip_y;
    double cio_s;
    double EO_sofa;
    AstroTime_t atime;
    iauASTROM a_sofa;
    double sun_to_source[3];  //Sun to source (unit vector, used for gravitational light deflection)
    double last_nutation_update;
    double nutation_update_interval;
    double last_jpl_update;
    double jpl_update_interval;
    bool debug;
    bool interpolate;

    int compute_times(double ctime);

    void get_nutations();

    void update_sofa(double earth_pv[6], double earth_ph[6]);

    int get_source_position(Object source, double ctime, double *ra_ICRS, double *dec_ICRS );

    void ICRS_to_CIRS(double ra_ICRS, double dec_ICRS, double *ra_CIRS, double *dec_CIRS);

  public:
    BPoint(char *ephem_path, double longitude, double lat, double height,
           ClockType clock=CLK_UT1, bool interp = true, bool dbg = false);

   ~BPoint() {};

    void set_earth_orientation(double dut1 = 0.0, double xp = 0.0, double yp = 0.0)
    {
      atime.dut1 = dut1; // seconds
      xpole = xp * DAS2R; // arcsec -> radians
      ypole = yp * DAS2R; // arcsec -> radians
    }

    void reprd ( char* s, double ra, double dc );

    void focal_plane_to_sky(
      int ndets, double det_dx[], double det_dy[], double det_pol_angle[],
      int nsamples, double x_ac[], double y_ac[], double rot_ac[],
      int flag,
      double x_det[], double y_det[], double rot_det[] );

    void recenter_focal_plane_to_coords(
      int ndets, int nsamples,
      // Detector coordinates, in some coordinate system, pa is parallactic angle.
      double det_lat[], double det_lon[], double det_pa[],
      // Re-center focal plane to these coordinates, pa is parallactic angle.
      double source_lat[], double source_lon[], double source_pa[],
      // Determines treatment of coordinates (0 = az/el, 1 = ra/dec)
      int flag,
      // Output coordinates
      double bearing[], double dist[], double pa[] );

    int compute_source(
      Object source, int nsamples, double ctime[],
      double az_src[], double alt_src[] );

    int recenter_focal_plane_to_source(
      Object source,
      int nsamples,
      double ctime[], double az_ac[], double alt_ac[], double rot_ac[],
      int ndets,
      double dx[], double dy[], double det_pol_angle[],
      double fp_dist[], double fp_bear[], double fp_rot[] );

    int recenter_focal_plane_to_sun(
      int nsamples,
      double ctime[], double az_ac[], double alt_ac[], double rot_ac[],
      int ndets,
      double dx[], double dy[], double det_pol_angle[],
      double fp_dist[], double fp_bear[], double fp_rot[] )
      {
        return recenter_focal_plane_to_source( Sun, nsamples, ctime, az_ac, alt_ac, rot_ac,
                                           ndets, dx, dy, det_pol_angle, fp_dist, fp_bear, fp_rot);
      };

    int recenter_focal_plane_to_moon(
      int nsamples,
      double ctime[], double az_ac[], double alt_ac[], double rot_ac[],
      int ndets,
      double dx[], double dy[], double det_pol_angle[],
      double fp_dist[], double fp_bear[], double fp_rot[] )
      {
        return recenter_focal_plane_to_source( GeoCMoon, nsamples, ctime, az_ac, alt_ac, rot_ac,
                                           ndets, dx, dy, det_pol_angle, fp_dist, fp_bear, fp_rot);
      };

    int recenter_focal_plane_to_venus(
      int nsamples,
      double ctime[], double az_ac[], double alt_ac[], double rot_ac[],
      int ndets,
      double dx[], double dy[], double det_pol_angle[],
      double fp_dist[], double fp_bear[], double fp_rot[] )
      {
        return recenter_focal_plane_to_source( Venus, nsamples, ctime, az_ac, alt_ac, rot_ac,
                                           ndets, dx, dy, det_pol_angle, fp_dist, fp_bear, fp_rot);
      };

};

#endif
