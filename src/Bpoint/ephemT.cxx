
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "jpl_eph.h"

FILE *test_stream = NULL;

int open_test_file (char *file_nm)
{
    if (test_stream != NULL) {
        printf ("open_test_file: Test file already open\n");
        return 0;
    }
    test_stream = fopen(file_nm, "rb");
    if (test_stream == NULL) {
        printf ("open_test_file: Cannot open file '%s'\n", file_nm);
        return 0;
    }
    // skip the identifying lines
    char first_word[100];
    strcpy(first_word, "");
    int safety = 50;
    while (strcmp(first_word, "EOT") && safety-- > 0) {
        fscanf(test_stream, "%s", first_word);
    }
    return 1;
}

void close_test_file()
{
    if (test_stream != NULL) {
        fclose(test_stream);
    }
}

// the numbering convention for test target and reference is:
//  1 = Mercury           8 = Neptune
//  2 = Venus             9 = Pluto
//  3 = Earth            10 = Moon
//  4 = Mars             11 = Sun
//  5 = Jupiter          12 = Solar-System Barycenter
//  6 = Saturn           13 = Earth-Moon Barycenter
//  7 = Uranus           14 = Nutations (Longitude and Obliq)
Object get_object(int num)
{
    switch (num) {
        case 1:
            return Mercury;
        case 2:
            return Venus;
        case 3:
            return numObj;
        case 4:
            return Mars;
        case 5:
            return Jupiter;
        case 6:
            return Saturn;
        case 7:
            return Uranus;
        case 8:
            return Neptune;
        case 9:
            return Pluto;
        case 10:
            return numObj;
        case 11:
            return Sun;
        case 12:
            return numObj;
        case 13:
            return EM_Bary;
        case 14:
            return Nutation;
        case 15:
            return Libration;
        default:
            return numObj;
    }
}

int ephemT(char * ephfile = 0, char * testfile = 0)
{
    if ((ephfile == 0) || (testfile == 0)) {
        printf ("Usage: ephemT ephemeris_file_name test_comparison_file\n");
        return 1;
    }
    if (!open_ephemeris(ephfile)) {
        return 1;
    }
    if (!open_test_file(testfile)) {
        return 1;
    }
    char eph_number[10];
    char date[30];
    double jd;
    int target, reference, coord_num;
    double answer;
    int safety = 1000;
    int skip;
    double tptr[6], rptr[6], targ_posn[6], ref_posn[6];
    while ((fscanf(test_stream, "%s %s %lf %d %d %d %lf",
                   eph_number, date, &jd, &target, &reference, &coord_num,
                   &answer) != EOF) && safety-- > 0) {
        printf("%s %s %6.2f %3d %3d %3d %20.13e ", eph_number, date, jd, target,
               reference, coord_num, answer);
	double au;
	if (!strcmp(eph_number, "200")) {
	    au = 0.149597870659999996e+09;
	} else if (!strcmp(eph_number, "403")) {
	    au = 0.149597870691000015e+09;
	} else if (!strcmp(eph_number, "405")) {
	    au = 0.149597870691000015e+09;
	} else {
	    printf("Ephemeris number not recognized\n");
	}
        Object targ = get_object(target);
        Object ref  = get_object(reference);
        skip = 0;
        if (target == 3) {
            if (!get_earth_posn_vel(targ_posn, SS_Barycentric, jd, 0.0))
		return 1;
        } else if (target == 10) {
            if (!get_moon_posn_vel(targ_posn, SS_Barycentric, jd, 0.0))
		return 1;
        } else if (target == 11) {
            if (!get_sun_posn_vel(targ_posn, SS_Barycentric, jd, 0.0))
		return 1;
        } else if (target == 12) {
	    int i;
	    for (i = 0; i < 6; i++) {
		targ_posn[i] = 0.0;
	    }
	} else if ((target < 1) || (target > 13)) skip = 1;
          else {
            if (!get_planet_posn_vel(targ_posn, targ, SS_Barycentric, jd, 0.0))
		return 1;
        }
	if (reference == 3) {
	    if (!get_earth_posn_vel(ref_posn, SS_Barycentric, jd, 0.0))
		return 1;
	} else if (reference == 10) {
	    if (!get_moon_posn_vel(ref_posn, SS_Barycentric, jd, 0.0))
		return 1;
	} else if (reference == 11) {
	    if (!get_sun_posn_vel(ref_posn, SS_Barycentric, jd, 0.0))
		return 1;
	} else if (reference == 12) {
	    int i;
	    for (i = 0; i < 6; i++) {
		ref_posn[i] = 0.0;
	    }
	} else if ((reference < 1) || (reference > 13)) skip = 1;
	  else {
	    if (!get_planet_posn_vel(ref_posn, ref, SS_Barycentric, jd, 0.0))
		return 1;
	}
/*
        for (i = 0; i < 6; i++) {
	    printf("%10.6f", targ_posn[i] / au);
	}
	printf("\n");
	for (i = 0; i < 6; i++) {
	    printf("%10.6f", ref_posn[i] / au);
	}
	printf("\n");
	for (i = 0; i < 6; i++) {
	    printf("%10.6f", (targ_posn[i] - ref_posn[i]) / au);
	}
	printf("\n");
*/
	int i = coord_num - 1;
        if((target == 14) && (reference == 0)) {
          if(!get_nutation(targ_posn, jd, 0.0))return(1);
          double diff = targ_posn[i] - answer;
	    if (fabs(diff) > 1.0e-13) {
	      printf ("----------> Difference > 1.0e-13 <----------\n");
          }
          printf (" Difference = %10.8e\n", diff);
        }

        else if(skip)printf(" Record skipped \n");

        else {
	double diff = (targ_posn[i] - ref_posn[i]) / au - answer;
	if (fabs(diff) > 1.0e-13) {
	    printf ("----------> Difference > 1.0e-13 <----------\n");
	}
	printf (" Difference = %10.8e\n", diff);
	}
    }
    return 0;
}

int main( int argc, char *argv[] )
{

    ephemT( argv[1], argv[2] );

    return 0;
}
