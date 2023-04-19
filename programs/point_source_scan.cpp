#include <Sky/sky.hpp>
#include <Polbeam/polbeam.hpp>
#include <Scanning/scanning.hpp>
#include <Mapper/mapper.hpp>
#include <Convolution/convolution_engine.h>

// timing
#include <chrono>
#include <unistd.h>

// utility
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (1.0 / DEG2RAD)
// sky, beam and map resolution parameters
#define NSIDE_SKY (256)
#define NSIDE_BEAM (1024)
#define NSIDE_MAP (256)
// beam parameters
#define FWHMX_DEG (3.0)
#define FWHMY_DEG (3.0)
#define PHI0_DEG (0.0)
// scanning parameters
#define NRA (128)
#define NDEC (128)
#define NPA (3)
#define NSAMP (NRA * NDEC * NPA)
#define RA0_DEG (45.0)
#define DEC0_DEG (0)
#define PA0_DEG (0.0)
#define DRA_DEG (15)
#define DDEC_DEG (15)
#define DPA_DEG (45.0)
// in radians, please
#define RA0 (RA0_DEG * DEG2RAD)
#define DEC0 (DEC0_DEG * DEG2RAD)
#define PA0 (PA0_DEG * DEG2RAD)
#define DRA (DRA_DEG * DEG2RAD)
#define DDEC (DDEC_DEG * DEG2RAD)
#define DPA (DPA_DEG * DEG2RAD)

int main(void)
{
    int s;
    float data_a[NRA * NDEC * NPA];
    float data_b[NRA * NDEC * NPA];

    Scanning scan;
    Sky sky(NSIDE_SKY);
    PolBeam beam(NSIDE_BEAM, 0.2);
    Mapper mapper(NSIDE_MAP);
    ConvolutionEngine conv(NSIDE_SKY, NSIDE_BEAM, NSAMP, 0.2);

    // make beams of detector a and b to be gaussian, no cross-pol
    beam.allocate_buffers();
    beam.make_unpol_gaussian_elliptical_beam('a', FWHMX_DEG, FWHMY_DEG, PHI0_DEG);
    beam.make_unpol_gaussian_elliptical_beam('b', FWHMX_DEG, FWHMY_DEG, PHI0_DEG);
    beam.normalize(NSIDE_SKY);

    // make sky to be a point source located at RA = 45deg, dec = 0.0 and 
    // be polarized in Q.
    sky.allocate_buffers();
    sky.make_point_source_sky(RA0, DEC0, 1.0, 1.0, 0.0, 0.0);

    // make a raster scan around point source
    scan.make_raster_scan(NRA, NDEC, NPA, RA0, DRA, DEC0, DDEC, PA0, DPA);

    // setup convolution engine
    conv.allocate_host_buffers();
    conv.allocate_device_buffers();
    conv.calculate_sky_pixel_coordinates(&sky);
    conv.beam_to_cuspvec(&beam);
    conv.sky_to_cuspvec(&sky);
    conv.create_matrix();
    
    auto t1s = std::chrono::steady_clock::now();
    auto t1e = std::chrono::steady_clock::now();
    double time_fill = 0.0;
    double time_transfer = 0.0;
    double time_conv = 0.0;
    double time_wall = 0.0;

    auto start = std::chrono::steady_clock::now();
    // execute convolution
    // first step is outside the loop to allow better concurrency
    s = 0;
    while(s < scan.size())
    {
        t1s = std::chrono::steady_clock::now();
        conv.fill_matrix(&sky, &beam, 
            scan.get_ra_ptr()[s], scan.get_dec_ptr()[s], scan.get_pa_ptr()[s]);
        t1e = std::chrono::steady_clock::now();
        time_fill += std::chrono::duration_cast<std::chrono::microseconds>(t1e - t1s).count();
        
        t1s = std::chrono::steady_clock::now();
        conv.exec_transfer();
        t1e = std::chrono::steady_clock::now();
        time_transfer += std::chrono::duration_cast<std::chrono::microseconds>(t1e - t1s).count();

        t1s = std::chrono::steady_clock::now();
        conv.exec_single_convolution_step(s);
        t1e = std::chrono::steady_clock::now();
        time_conv += std::chrono::duration_cast<std::chrono::microseconds>(t1e - t1s).count();

        s = s + 1;
    }
    conv.sync();
    // transfer data back to host, as TOD (i + q + u)
    conv.iqu_to_tod(data_a, data_b);
    auto end = std::chrono::steady_clock::now();
    time_wall = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // report

    std::cerr << "Time filling matrix: " << time_fill / 1000.0 << " ms" << std::endl;
    std::cerr << "Time transfering matrix: " << time_transfer / 1000.0 << " ms" << std::endl;
    std::cerr << "Time using matrix: " << time_conv / 1000.0 << " ms" << std::endl;
    std::cerr << "Time not accounted for: " << (time_fill + time_transfer + time_conv - time_wall) / 1000.0 << " ms" << std::endl;
    std::cerr << "[INFO] Computed " << NRA * NDEC * NPA << " samples in " << time_wall / 1000.0 << " ms ";
    std::cerr << "(" << (NRA * NDEC * NPA) / (time_wall / 1E6) << " samples/sec)" << std::endl;

    // transform results to a map
    mapper.accumulate_data(scan.size(), 
        scan.get_ra_ptr(), scan.get_dec_ptr(), scan.get_pa_ptr(), 0.0, data_a);
    // detector b has a polarization angle of 90 degrees.
    mapper.accumulate_data(scan.size(), 
        scan.get_ra_ptr(), scan.get_dec_ptr(), scan.get_pa_ptr(), 90.0 * DEG2RAD, data_b);
    mapper.solve_map();

    // output map to stdout
    float f = (1.0 * NSIDE_SKY) / (1.0 * NSIDE_MAP);
    f = f * f;
    for(s = 0; s < mapper.size(); s++)
    {
        std::cout << mapper.get_stokes_I()[s] * f << " ";
        std::cout << mapper.get_stokes_Q()[s] * f << " ";
        std::cout << mapper.get_stokes_U()[s] * f << " ";
        std::cout << mapper.get_hitmap()[s] << std::endl;
    }
    
    return 0;
}