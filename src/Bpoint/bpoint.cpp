#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "BPoint.h"
//#include "ElapsedTime.h"

// Defining HIGH_PRECISION true will reduce the error from around 50 mas to
// less than 0.1 mas at the cost of about a 10% increase in execution time.
#define HIGH_PRECISION false

BPoint::BPoint(char *ephem_path, double longitude, double lat, double height,
               ClockType clock, bool interp, bool dbg)
{
    jpl_ephem_path = ephem_path;
    site_longitude = longitude; // radians
    site_latitude = lat; // radians
    site_height = height; // meters
    atime.clock = clock;
    atime.dut1 = 0.0;
    xpole = 0.0;
    ypole = 0.0;
    // Update precession/nutations every 60 seconds, which is *overkill*
    nutation_update_interval = 60.0;
    // Update source coordinates every 1 seconds, which is *overkill*
    jpl_update_interval = 1.0;
    interpolate = interp;
    debug = dbg;
};

void BPoint::reprd ( char* s, double ra, double dc )
{
    char pm;
    int i[4];
    printf ( "%25s", s );
    iauA2tf ( 7, ra, &pm, i );
    printf ( " %2.2d %2.2d %2.2d.%7.7d", i[0],i[1],i[2],i[3] );
    iauA2af ( 6, dc, &pm, i );
    printf ( " %c%2.2d %2.2d %2.2d.%6.6d\n", pm, i[0],i[1],i[2],i[3] );
}

int BPoint::compute_times(double ctime)
/*
   This will compute the exact same values of UT1 and TAI as SOFA's method during
   a leap second event, without all of the compute intensive rigmarole that SOFA
   goes through. If we are interpolating UT1-UTC, the interpolation will be off
   by TAI-UTC if using a TAI clock, but that will introduce an error that is much
   less than the uncertainty in UT1-UTC published in IERS Bulletin B.
*/
{
    double d1, d2, w, dtai;
    int stat, iy, imo, id;

    // This is SOFA's recommended method for forming the two part Julian date.
    d1 = DJ00;
    d2 = (ctime - UNIX_JD2000) / DAYSEC;

     /*
        Look up TAI-UTC. If using a non-UTC clock, this will happen
        before of after the leap second event, but that makes absolutely
        no difference to UT1 and TAI.
    */
    stat = iauJd2cal(d1, d2, &iy, &imo, &id, &w);
    if (stat != 0)
    {
        fprintf( stderr, "Error getting UTC date. Aborting. \n" );
        return 0;
    }
    stat = iauDat ( iy, imo, id, 0.0, &dtai);
    if (stat != 0)
    {
        fprintf( stderr, "Error getting TAI-UTC for %04d-%02d-%02d Aborting. \n", iy, imo, id );
        return 0;
    }
    // We should look up UT1-UTC at this time. But for now just modify the
    // given value if a leap second occurred.
    if((atime.dtai > 0.0) && (atime.dtai != dtai))
    {
        atime.dut1 += (dtai - atime.dtai);
        //printf("%d-%d-%d %lf %lf %lf\n", iy, imo, id, w, dtai, atime.dut1);
    }
    atime.dtai = dtai;

    atime.jd_ut1[0] = d1;
    atime.jd_tt[0]  = d1;
    atime.jd_tt[1]  = TTMTAI / DAYSEC;

    switch(atime.clock)
    {
        case CLK_UTC:
            atime.jd_ut1[1] =  d2 + atime.dut1 / DAYSEC;
            atime.jd_tt[1] += (d2 + atime.dtai / DAYSEC);
            break;
        case CLK_UT1:
            atime.jd_ut1[1] =  d2;
            atime.jd_tt[1] += (d2 + (atime.dtai - atime.dut1) / DAYSEC);
            break;
        case CLK_TAI:
            atime.jd_ut1[1] =  d2 + (atime.dut1 - atime.dtai) / DAYSEC;
            atime.jd_tt[1] +=  d2;
            break;
        case CLK_GPS:
            atime.jd_ut1[1] =  d2 + (TAIMGPS + atime.dut1 - atime.dtai) / DAYSEC;
            atime.jd_tt[1] += (d2 +  TAIMGPS / DAYSEC);
            break;
        default:
            printf("Unknown clock type. Aborting. \n");
            return 0;
    }

    return 1;
}

#if HIGH_PRECISION
void BPoint::get_nutations()
// This is SOFA's S00a with the equation of the origins added.
{
    double eqbpn[3][3];

    // Look ahead by half of the nutation_update_interval
    double jd_tt2 = atime.jd_tt[1] + nutation_update_interval / (2.0 * DAYSEC);

    /* Form the equinox based BPN matrix, IAU 2006/2000A. */
    iauPnm06a(atime.jd_tt[0], jd_tt2, eqbpn);

    /* Extract CIP X,Y. */
    iauBpn2xy(eqbpn, &cip_x, &cip_y);

    /* Obtain CIO locator s. */
    cio_s = iauS06(atime.jd_tt[0], jd_tt2, cip_x, cip_y);

    /* Equation of the origins. */
    // Only needed if we want LST and apparent RA wrt the true equator and equinox of date.
    if(debug)EO_sofa = iauEors(eqbpn, cio_s);

}

#else
void BPoint::get_nutations()
// This is SOFA's S00b with the equation of the origins added. ~1 mas inaccuracy.
{
    double eqbpn[3][3];

    // Look ahead by half of the nutation_update_interval
    double jd_tt2 = atime.jd_tt[1] + nutation_update_interval / (2.0 * DAYSEC);

    /* Form the equinox based BPN matrix, IAU 2000/2000B. */
    iauPnm00b(atime.jd_tt[0], jd_tt2, eqbpn);

    /* Extract CIP X,Y. */
    iauBpn2xy(eqbpn, &cip_x, &cip_y);

    /* Obtain CIO locator s. */
    cio_s = iauS00(atime.jd_tt[0], jd_tt2, cip_x, cip_y);

    /* Equation of the origins. */
    // Only needed if we want LST and apparent RA wrt the true equator and equinox of date.
    if(debug)EO_sofa = iauEors(eqbpn, cio_s);

}
#endif

void BPoint::update_sofa(double earth_pv[6], double earth_ph[6])
{
    double ebpv[2][3], ehp[3];
    int i;

    /* convert from km, km/day to au, au/day */
    for(i = 0; i < 3; i++)
    {
        ebpv[0][i] = earth_pv[i] * 1000.0 / DAU;
        ebpv[1][i] = earth_pv[i + 3] * 1000.0 / DAU;
        ehp[i]     = earth_ph[i] * 1000.0 / DAU;
    }

    /* TIO locator s'. */
    double sp = 0.0;

    /* Refraction constants A and B. */
    double refa = 0.0;
    double refb = 0.0;

    /* Polar motion */
    double xp = xpole;
    double yp = ypole;

    /* Earth rotation angle. */
    double theta = iauEra00(atime.jd_ut1[0], atime.jd_ut1[1]);

    /* Compute the star-independent astrometry parameters. */
    iauApco(atime.jd_tt[0], atime.jd_tt[1], ebpv, ehp, cip_x, cip_y, cio_s, theta,
           site_longitude, site_latitude, site_height,
           xp, yp, sp, refa, refb, &a_sofa);

}

// Get ICRS equatorial coordinates of source
int BPoint::get_source_position( Object source, double ctime, double *ra_ICRS, double *dec_ICRS )
{
    double source_pos[6], earth_pv[6], source_ph[6];
    double dist;
    int stat, i;
    double earth_ph[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if( !get_earth_posn_vel( earth_pv, SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1] ) )
    {
        fprintf( stderr, "Error getting the position and velocity of the Earth. Aborting.\n" );
        return 0;
    }
#if HIGH_PRECISION
    // This is for the optional gravitaional light deflection by the Sun.
    if( !get_sun_posn_vel( earth_ph, SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1], 0.0 ) )
    {
        fprintf( stderr, "Error getting the position of the Sun. Aborting.\n" );
        return 0;
    }
    if( !get_sun_posn_vel( source_ph, SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1] - tlight, 0.0 ) )
    {
        fprintf( stderr, "Error getting the light delayed position of the Sun. Aborting.\n" );
        return 0;
    }
#endif
    switch( source )
    {
        case Sun:
            if( !get_sun_posn_vel( source_pos, SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1] - tlight, 0.0 ) )
            {
                fprintf( stderr, "Error getting the position of the Sun. Aborting.\n" );
                return 0;
            }
            break;
        case GeoCMoon:
            if( !get_moon_posn_vel( source_pos, SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1] - tlight, 0.0 ) )
            {
                fprintf( stderr, "Error getting the position of the Moon. Aborting.\n" );
                return 0;
            }
            break;
        case Mercury:
        case Venus:
        case Mars:
        case Jupiter:
        case Saturn:
        case Uranus:
        case Neptune:
        case Pluto:
            if( !get_planet_posn_vel( source_pos, source,  SS_Barycentric, atime.jd_tt[0], atime.jd_tt[1] - tlight, 0.0 ) )
            {
                fprintf( stderr, "Error getting the position of Planet. Aborting.\n" );
                return 0;
            }
            break;
        default:
            fprintf( stderr, "Unknown source. Aborting.\n" );
            return 0;
    }

    if( ctime >= last_nutation_update + nutation_update_interval )
    {
        get_nutations();
        last_nutation_update = ctime;
    }

#if HIGH_PRECISION
    for (i=0; i<3; i++) {
      // Heliocentric
      earth_ph[i]  = earth_pv[i]   - earth_ph[i]; // km
      // Heliocentric light delayed
      source_ph[i] = source_pos[i] - source_ph[i]; // km
    }
    /* Heliocentric direction and distance (unit vector and km). */
    iauPn(source_ph, &dist, sun_to_source);
#endif
    update_sofa(earth_pv, earth_ph);
    /*
    printf("heliocentic vectors: source (%lf, %lf, %lf) earth (%lf, %lf, %lf) \n", source_ph[0], source_ph[1], source_ph[2],
                                                                                   earth_ph[0], earth_ph[1], earth_ph[2]);

    printf("       unit vectors: source (%lf, %lf, %lf) earth (%lf, %lf, %lf) \n", sun_to_source[0], sun_to_source[1], sun_to_source[2],
                                                                                   a_sofa.eh[0], a_sofa.eh[1], a_sofa.eh[2]);
    */
    dist = 0.0;
    for (i=0; i<3; i++) {
      // Topocentric
      source_pos[i] -= (a_sofa.eb[i] * DAU / 1000.0); // km
      dist  += source_pos[i] * source_pos[i];
    }
    dist = sqrt(dist);
    // Light time to Source (days)
    tlight = dist / (C_LIGHT * DAYSEC);
    //printf("tlight %lf distance %lf \n", tlight * 1440.0, dist);
    iauC2s ( source_pos, ra_ICRS, dec_ICRS);

    *ra_ICRS = iauAnp(*ra_ICRS);

    return 1;
}

void BPoint::ICRS_to_CIRS(double ra_ICRS, double dec_ICRS, double *ra_CIRS, double *dec_CIRS)

/* This is SOFA's atciqz with gravitational deflection of light from the Sun optionally
 * omitted.
 *
 * SOFA's atciqz handles gravitational deflection of light incorrectly for solar system bodies.
 * This is optionally corrected here by computing the heliocentric position (including light
 * time for the Sun) of the target body and calling the SOFA function ld() directly with it.
 * But this option would only be useful for an instrument with a much smaller beam capable
 * of targeting a planet close to superior conjunction with the Sun. At 10 degrees from the
 * Sun, gravitational deflection is about 50 mas.
 *
 */
{
   double pco[3], pnat[3], ppr[3], pi[3], w;
   double em2, dlim;

/* BCRS coordinate direction (unit vector). */
   iauS2c(ra_ICRS, dec_ICRS, pco);

#if HIGH_PRECISION
/* Light deflection by the Sun, giving BCRS natural direction. */
/* Deflection limiter (smaller for distant observers). */
   em2 = a_sofa.em * a_sofa.em;
   dlim = 1e-6 / (em2 > 1.0 ? em2 : 1.0);

/* Apply the deflection. */
   iauLd(1.0, pco, sun_to_source, a_sofa.eh, a_sofa.em, dlim, pnat);

/* Aberration, giving GCRS proper direction. */
   iauAb(pnat, a_sofa.v, a_sofa.em, a_sofa.bm1, ppr);
#else
/* Aberration, giving GCRS proper direction. */
   iauAb(pco, a_sofa.v, 1.0, a_sofa.bm1, ppr);
#endif

/* Bias-precession-nutation, giving CIRS proper direction. */
   iauRxp(a_sofa.bpn, ppr, pi);

/* CIRS RA,Dec. */
   iauC2s(pi, &w, dec_CIRS);
   *ra_CIRS = iauAnp(w);
}

// Get apparent horizontal coordinates of source
int BPoint::compute_source(
    Object source, int nsamples, double ctime[],
    double az_src[], double alt_src[] )
{
    long i;

    // Initialize some stuff
    tlight = 0.0;
    last_nutation_update = 0.0;
    last_jpl_update = 0.0;
    atime.dtai = 0.0;
    // Sanity check
    if((ctime[1] - ctime[0]) > jpl_update_interval)interpolate = false;

    // We need the az/alt of the source for the ctime stream
    // assumes the ctime is in UT1 scale!

    // First, compute ra/dec of the source using JPL ephem

    // Open JPL Ephemeris file
    if( !open_ephemeris( jpl_ephem_path ) )
    {
        fprintf( stderr, "Could not open ephemeris file. Aborting\n" );
        return 0;
    }
    // Mainly for comparison with JPL Horizons/Skyfield
    if(debug)printf("UT,LST,ICRS_RA,ICRS_DEC,RA,DEC,AZ,EL,DIST,UTC,UT1,TT \n");

    /* Interpolation stuff. This will smooth out the saw tooth ramp in Az/El between
     * lookups. The maximum amplitude of this is only 0.5 arc seconds per second
     * on the Moon, so this is really unnecessary unless jpl_update_interval is
     * large, you're using this algorithm to control an optical telescope or you're
     * just an incurable perfectionist (like me). That said, it adds a negligible
     * amount of compute time, so why not.
     */
    double ra_ICRS[3], dec_ICRS[3], ra_CIRS[3], dec_CIRS[3], t[2], frac;
    double look_ahead = jpl_update_interval / DAYSEC;

    for( i=0; i < nsamples; i++ )
    {
        if(!compute_times(ctime[i])) return 0;

        // Restart interpolation if ctime jumps
        if((i == 0) || (interpolate && (abs(ctime[i] - ctime[i-1]) > jpl_update_interval)))
        {
            if(i == 0)
            {
               // Do this to initialize light time (It will get more accurate as we iterate on it)
               if(!get_source_position(source, ctime[i], &ra_ICRS[1], &dec_ICRS[1] )) return 0;
            }
            // If interpolating, get initial positions
            if(interpolate)
            {
                t[1] = ctime[i];
                if(!get_source_position(source, ctime[i], &ra_ICRS[1], &dec_ICRS[1] )) return 0;
                /* Transform from ICRS to CIRS*/
                ICRS_to_CIRS( ra_ICRS[1], dec_ICRS[1], &ra_CIRS[1], &dec_CIRS[1] );
            }
        }

        // Get vector pointing from the Earth to the source
        if( ctime[i] >= last_jpl_update + jpl_update_interval )
        {
            last_jpl_update = ctime[i];
            if(interpolate)
            {
                t[0] = t[1];
                t[1] = ctime[i] + jpl_update_interval;
                 ra_ICRS[0] =  ra_ICRS[1];
                dec_ICRS[0] = dec_ICRS[1];
                 ra_CIRS[0] =  ra_CIRS[1];
                dec_CIRS[0] = dec_CIRS[1];
                atime.jd_ut1[1] += look_ahead;
                atime.jd_tt[1]  += look_ahead;
            }
            if(!get_source_position(source, ctime[i], &ra_ICRS[1], &dec_ICRS[1] )) return 0;
#if HIGH_PRECISION
            // Improved estimate of light travel time.
            if(!get_source_position(source, ctime[i], &ra_ICRS[1], &dec_ICRS[1] )) return 0;
#endif
            /* Transform from ICRS to CIRS*/
            ICRS_to_CIRS( ra_ICRS[1], dec_ICRS[1], &ra_CIRS[1], &dec_CIRS[1] );
            // Reset time
            if(interpolate)
            {
                atime.jd_ut1[1] -= look_ahead;
                atime.jd_tt[1]  -= look_ahead;
            }
        }

        if(interpolate)
        {
            frac = (ctime[i] - t[0]) / (t[1] - t[0]);
            ra_CIRS[2]  = ra_CIRS[0]  + frac * iauAnp(ra_CIRS[1] - ra_CIRS[0]);
            dec_CIRS[2] = dec_CIRS[0] + frac * (dec_CIRS[1] - dec_CIRS[0]);
        }
        else
        {
            ra_CIRS[2]  = ra_CIRS[1];
            dec_CIRS[2] = dec_CIRS[1];
        }

        // Update the local Earth Rotation Angle
        iauAper13(atime.jd_ut1[0], atime.jd_ut1[1] , &a_sofa );

        // Convert CIRS coordinates of the source to horizontal coordinates at the CLASS observatory
        /*Transform from CIRS to observed az/alt*/
        /* Note that "observed" ra, dec and ha here are wrt the ITRS (which is rotated through polar
         * motion) and include refraction, so they aren't topcentric apparent place in the classical
         * sense at all. Also note that diurnal aberration is added here in the case of geocentric
         * CIRS coordinates (apio), whereas diurnal aberration is added to the CIRS coordinates in
         * the case of topocentric CIRS coordinates (apco). This appears to be a logical inconsistency
         * on the part of SOFA. Since we are using topocentric CIRS coordinates with diurnal aberration
         * included, classical topocentric apparent place can be derived directly from the topocentric
         * CIRS coordinates.
         */
        double az, za, ha, obs_dec, obs_ra;
        iauAtioq( ra_CIRS[2], dec_CIRS[2], &a_sofa, &az, &za, &ha, &obs_dec, &obs_ra );

         az_src[i] = az;
        alt_src[i] = M_PI_2 - za;

        if(debug)
        {
            /* Translate CIRS RA to actual apparent RA wrt the true equinox and equator of date
             * for comparison with JPL Horizons and <begin_rant> ancient astronomers like me who believe
             * that this CIO, ERA and CIRS RA are nothing but convenient mathematical constructs
             * adopted by the IAU that, while having computational merit, have little to do with observable
             * reality and should not be foisted upon the astronomical community as a viable replacement
             * for apparent place and local sidereal time. Some believe that apparent place and LST will
             * go the way of the sextant in the age of GPS. I hope that isn't true. There is true beauty
             * in the concept of the Sun crossing the equator at 0 hours apparent RA on the day of the
             * Vernal equinox. jmo </end_rant>
             */

            // Get longitude correction for polar motion
            double dlong = a_sofa.ypl * tan( site_latitude);
            //printf("dlong %lf \n", dlong * DR2AS);
            double ra  = iauAnp(ra_CIRS[2] - EO_sofa);
            double dec = dec_CIRS[2];
            double lst = iauAnp(a_sofa.eral + dlong - EO_sofa) * 12.0 / M_PI;

            if(interpolate)
            {
                ra_ICRS[2]  = ra_ICRS[0]  + frac * iauAnp(ra_ICRS[1] - ra_ICRS[0]);
                dec_ICRS[2] = dec_ICRS[0] + frac * (dec_ICRS[1] - dec_ICRS[0]);
            }
            else
            {
                ra_ICRS[2]  = ra_ICRS[1];
                dec_ICRS[2] = dec_ICRS[1];
            }

            double d1, d2;
            int stat, iy, imo, id, ihmsf[4];
            /*
               This returns a "special" version of d2 that has 86401 seconds per day on the
               day before a leap second. Need to encode it using iauD2dtf and then decode
               it to get to the normal d2. This encoding will also properly print UTC as
               23:59:60.ssssss during the leap second event if we are using a non-UTC clock.
               If we are using a UTC clock, leap seconds don't show up in UTC anyway. UT1
               and TT just advance by one second. Unless we are using an actual UTC timestream
               from an OS that can't do 23:59:60.xxx. Then 23:59:59 is simply repeated with
               an accompanying one second glitch in UT1 and TAI.
            */
            stat = iauTaiutc(atime.jd_tt[0], atime.jd_tt[1] - TTMTAI / DAYSEC, &d1, &d2);
            if (stat != 0)
            {
               fprintf( stderr, "Error getting UTC. Aborting. \n" );
               return 0;
            }
            stat = iauD2dtf("UTC", 6, d1, d2, &iy, &imo, &id, ihmsf);
            if (stat != 0)
            {
               fprintf( stderr, "Error encoding UTC. Aborting. \n" );
               return 0;
            }
            stat = iauCal2jd(iy, imo, id, &d1, &d2);
            if (stat != 0)
            {
                 fprintf( stderr, "Error decoding UTC date. Aborting. \n" );
                 return 0;
            }
            d2 += (d1  - DJ00);
            d2 += ((double) (60 * (60 * ihmsf[0] + ihmsf[1]) + ihmsf[2]) + (double) ihmsf[3] * 1.0e-6) / DAYSEC;

            char buf[30];
            // Format time for Excel
            sprintf(buf,"%4d-%02d-%02d %02d:%02d:%02d.%06d", iy, imo, id, ihmsf[0],
                        ihmsf[1], ihmsf[2], ihmsf[3]);

            // Mainly for comparison with JPL Horizons/Skyfield
            printf("%s,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.9lf,%.3lf,%.9lf,%.9lf,%.9lf\n",
                   buf, lst,
                   ra_ICRS[2] * RAD2DEG, dec_ICRS[2] * RAD2DEG,
                   ra * RAD2DEG, dec * RAD2DEG,
                   az_src[i] * RAD2DEG, alt_src[i] * RAD2DEG,
                   tlight * C_LIGHT * DAYSEC * 1000.0,
                   d2, atime.jd_ut1[1], atime.jd_tt[1]
                  );
       }
    }

    close_ephemeris();
    return 1;
}

/* Converts array from receiver coordinate system to sky. Perspective looking out to sky.*/
void BPoint::focal_plane_to_sky(
    int ndets, double det_dx[], double det_dy[], double det_pol_angle[],
    int nsamples, double x_ac[], double y_ac[], double rot_ac[],
    int flag,
    double x_det[], double y_det[], double rot_det[] )
{

/* Converts array from receiver coordinate system to sky. Perspective looking out to sky.
 *
 * Input:   det_dx (double): Receiver coordinate detector x offset from array center (positive right)
 *          det_dy (double): Receiver coordinate detector y offset from array center (positive up)
 *   det_pol_angle (double): Detector polarization angle ccw from receiver coordinate y axis
 *
 *
 * flag (int): = 0 Az/El coordinates requested
 *
 *        x_ac (double): Array center azimuth on the sky
 *        y_ac (double): Array center elevation on the sky
 *      rot_ac (double): Array center boresight angle (ccw from zenith)
 *
 * flag (int): = 1 Ra/Dec coordinates requested
 *
 *        x_ac (double): Array center right ascension on the sky
 *        y_ac (double): Array center declination on the sky
 *      rot_ac (double): Array center position angle on the sky
 *                       (parallactic ccw from NCP/CIP/CEP + boresight ccw from zenith)
 *
 * flag (int): = 2 Source relative coordinates requested (Source is at the pole)
 *
 *        x_ac (double): Array center bearing wrt source on the sky (ccw from zenith)
 *        y_ac (double): Array center latitude wrt source on the sky
 *      rot_ac (double): Array center position angle wrt source
                         (position angle of zenith ccw from source (parallactic angle equivalent)
                          + boresight ccw from zenith)
 *
 * Output: flag (int): = 0 Az/El coordinates requested
 *
 *        x_det (double): Detector azimuth on the sky
 *        y_det (double): Detector elevation on the sky
 *      rot_det (double): Detector position angle on the sky (ccw from zenith)
 *
 * Output: flag (int): = 1 Ra/Dec coordinates requested
 *
 *        x_det (double): Detector right ascension on the sky
 *        y_det (double): Detector declination on the sky
 *      rot_det (double): Detector position angle on the sky (ccw from NCP/CIP/CEP)
 *
 * Output: flag (int): = 2 Source relative coordinates requested (Source is at the pole)
 *
 *        x_det (double): Detector bearing wrt source on the sky (ccw from source)
 *                        Alternately, source bearing wrt detector (cw from receiver coordinate pole)
 *        y_det (double): Detector latitude wrt source on the sky
 *      rot_det (double): Detector position angle on the sky (ccw from source)
 *
 * Algorithm by Michael Brewer.
 */

    // Compute on-sky offsets and position angle of detectors
    #pragma omp parallel
    { // begin parallel region

    #pragma omp for
    for( int det=0; det < ndets; det++ ) {

      double dx      = det_dx[det];
      double cdx     = cos(dx);
      double sdx     = sin(dx);
      double cdy     = cos(det_dy[det]);
      double sdy     = sin(det_dy[det]);
      double polang  = det_pol_angle[det];
      // detector polar coordinates in receiver coordinate system
      double cr      = cdx * cdy;
      double sr      = sqrt((1.0 - cr) * (1.0 + cr));
      double alpha   = atan2(sdy, sdx * cdy);
      long det_index  = det * nsamples;

      for(int s = 0; s < nsamples; s++ ) {

        long index    = det_index + s;
        // gamma = detector -> array center -> zenith | pole | source
        double gamma = M_PI_2 - alpha - rot_ac[s];
        double cg    = cos(gamma);

        double sy    = sin(y_ac[s]);
        double cy    = cos(y_ac[s]);
        y_det[index] = asin(cr * sy + sr * cy * cg);

        double seta  = sin(rot_ac[s]);
        double ceta  = cos(rot_ac[s]);
        // l = distance from receiver coordinate pole to zenith | pole | source
        double cl    = cy * ceta;
        // delta = array center -> receiver coordinate pole -> zenith | pole | source
        // sin delta = cy * seta / sin l, cos delta = sy / sin l
        // eps = delta - dx = detector -> receiver coordinate pole -> zenith | pole | source
        // sin l * sin eps
        double sl_seps = cy * seta * cdx - sy * sdx;
        // sin l * cos eps
        double sl_ceps = sy * cdx + cy * seta * sdx;
        // Position angle of the detector
        // zenith | pole | source -> detector -> receiver coordinate pole
        double pa_det = atan2(sl_seps, cl * cdy - sl_ceps * sdy);
        // Polarization direction of the detector wrt zenith | pole | source
        rot_det[index] = pa_det + polang;

        if(flag == 2) {
          // receiver coordinate pole -> detector -> source (source to the East is positive)
          x_det[index] = -pa_det;
        }
        else {
          double sg    = sin(gamma);
          // array center -> zenith | pole -> detector
          double delta_xdet = atan2( sr * sg, cr * cy - sr * sy * cg);

          // Azimuth to the West, Ra to the East
          if(flag == 0)x_det[index]  = x_ac[s] + delta_xdet;
          else x_det[index]  = x_ac[s] - delta_xdet;
        }
      }
    }
    } // end parallel region

}

void BPoint::recenter_focal_plane_to_coords(
    int ndets, int nsamples,
    // Detector coordinates, in some coordinate system, pa is parallactic angle.
    double det_lat[], double det_lon[], double det_pa[],
    // Re-center focal plane to these coordinates, pa is parallactic angle.
    double source_lat[], double source_lon[], double source_pa[],
    // Determines treatment of coordinates (0 = az/el, 1 = ra/dec)
    int flag,
    // Output coordinates
    double bearing[], double dist[], double pa[]
)
/*   This function calculates the great circle distance between points at
 *   (source_lon, source_lat) and (det_lon, det_lat), the bearing (ccw positive)
 *   of (det_lon, det_lat) wrt (source_lon, source_lat) and the position angle (ccw positive)
 *   of (source_lon, source_lat) wrt to the pole of (det_lon, det_lat)
 *
 *   ex. (det_lon,det_lat) east of (source_lon, source_lat) bearing = 90
 *       (det_lon,det_lat) south of (source_lon, source_lat) bearing = 180
 *       (det_lon, det_lat) west of (source_lon, source_lat) bearing = -90
 *       (det_lon, det_lat) north of (source_lon, source_lat) bearing = 0
 */
{

    #pragma omp parallel
    {
    int det, sample;
    long index;
    double phi, theta, lat1, lon1, lat2, lon2;
    double delta, x, y, c;
    double slat1, clat1, slat2, clat2, sdel, cdel, psi;

    #pragma omp for
    for( det=0; det < ndets; det++ )
    {
        for( sample=0; sample < nsamples; sample++ )
        {
            index = det*nsamples + sample;
            // Convenience definitions
            lat2 = det_lat   [ index ];
            lon2 = det_lon   [ index ];

            lat1 = source_lat[ sample ];
            lon1 = source_lon[ sample ];
            slat1 = sin(lat1);
            clat1 = cos(lat1);
            slat2 = sin(lat2);
            clat2 = cos(lat2);
            // det to the East is positive
            // interpret lon as azimuth
            if(flag == 0)delta = lon1 - lon2;
            // interpret lon as ra (longitude)
            else delta = lon2 - lon1;
            sdel = sin(delta);
            cdel = cos(delta);

            // Compute angle Pole - Source - Det
            x     =  sdel * clat2;
            y     = clat1 * slat2 - slat1 * clat2 * cdel;
            phi   = atan2( x, y );
            // Compute angle Pole - Det - Source
            x     =  sdel * clat1;
            y     = clat2 * slat1 - slat2 * clat1 * cdel;
            psi   = atan2( x, y );
            if(flag == 1)
            {
                // Convert to angles wrt Zenith
                phi -= source_pa[ index ];
                psi += det_pa[ index ];
            }

            // Compute distance between Det and Source
            c = slat1 * slat2 + clat1 * clat2 * cdel;
            theta = acos(c);

            dist   [ index ] = theta;
            bearing[ index ] = phi;
            pa     [ index ] = psi;
        }
    }
    } // end parallel region
}

int BPoint::recenter_focal_plane_to_source(
    Object source,
    int nsamples,
    double ctime[], double az_ac[], double alt_ac[], double rot_ac[],
    int ndets,
    double dx[], double dy[], double det_pol_angle[],
    double fp_dist[], double fp_bear[], double fp_rot[] )
{
    long i;
    // Need scratch buffers
    double  *az_src  = (double*)malloc( sizeof(double) * nsamples );
    double *alt_src  = (double*)malloc( sizeof(double) * nsamples );
    //ElapsedTime_t btime, etime1, etime2;
    //ElapsedTime ET;

    //btime = ET.elapse(btime);

    // Compute horizontal coordinates for source
    if(!compute_source(source, nsamples, ctime, az_src, alt_src )) return 0;

    //etime1 = ET.elapse(btime);

    // Need more scratch
    double * ac_dist = alt_src;
    double * ac_bear = az_src;
    double * ac_pa   = (double*)malloc( sizeof(double) * nsamples );

    // Compute source centered position of array center in horizontal coordinates
    recenter_focal_plane_to_coords(
        1, nsamples,
        alt_ac, az_ac, (double *) NULL ,
        alt_src, az_src, (double *) NULL ,
        0,
        ac_bear, ac_dist, ac_pa );

    // printf("ArrayCenter: Bearing %lf Distance %lf Position Angle %lf\n", ac_bear[0] * RAD2DEG, ac_dist[0] * RAD2DEG, ac_pa[0] * RAD2DEG);
    // Compute distance, bearing and position angle of the focal plane given
    // boresight pointing and offsets.
    for( i=0; i < nsamples; i++ )
    {
        //printf("%lf %lf %lf \n", ac_dist[i] * RAD2DEG, ac_bear[i] * RAD2DEG, ac_pa[i] * RAD2DEG);
        // Add in boresight angle
        ac_pa[i] += rot_ac[i];
        // convert distance to latitude
        ac_dist[i] = M_PI_2 - ac_dist[i];
    }

    focal_plane_to_sky(
        ndets, dx, dy, det_pol_angle,
        nsamples, ac_bear, ac_dist, ac_pa,
        2,
        fp_bear, fp_dist, fp_rot );

    //etime2 = ET.elapse(btime);
    //printf("compute source: et = %lf sec\n", etime1.etime);
    //printf("recenter_focal_plane_to_source : et = %lf sec\n", etime2.etime);

    // Clean up
    free(  az_src );
    free(  alt_src );
    free(  ac_pa );
    return 1;
}
