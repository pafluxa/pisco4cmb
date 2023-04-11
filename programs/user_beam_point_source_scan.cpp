#include <Sky/sky.hpp>
#include <Polbeam/polbeam.hpp>
#include <Scanning/scanning.hpp>
#include <Mapper/mapper.hpp>
#include <Convolution/convolution_engine.h>

// timing
#include <chrono>
#include <unistd.h>

// json parsing
#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

// utility
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (1.0 / DEG2RAD)

int main(int argc, char* argv[])
{
    // beam parameters
    int nside_beam;
    std::string det_a_beam_data;
    std::string det_b_beam_data;
    // sky parameters
    int nside_sky;
    // output map parameters
    int nside_output_map;
    std::string output_map_path;
    // scanning parameters
    int nsamp;
    // Right Ascention
    int nra;
    float ra0;
    float dra;
    // - declination
    int ndec;
    float dec0;
    float ddec;
    // - position angle
    int npa;
    float pa0;
    float dpa;

    // other variables
    int s;
    float *data_a;
    float *data_b;

    // Argument parsing
    if(argc != 2) {
        std::cerr << "Usage: user_beam_point_source_.x <JSON configuration file>" << std::endl;
        return 1;
    }

    // parse configuration file
    std::ifstream configfile(argv[1]);
    json config = json::parse(configfile);
    // sky parameters
    config.at("nside_sky").get_to(nside_sky);
    // beam parameters
    config.at("nside_beam").get_to(nside_beam);
    config.at("beam_detector_a").get_to(det_a_beam_data);
    config.at("beam_detector_b").get_to(det_b_beam_data);
    // output map
    config.at("nside_output_map").get_to(nside_output_map);
    config.at("output_map_path").get_to(output_map_path);
    // scanning
    config.at("ra0").get_to(ra0);
    config.at("dra").get_to(dra);
    config.at("npoints_ra").get_to(nra);
    config.at("dec0").get_to(dec0);
    config.at("ddec").get_to(ddec);
    config.at("npoints_dec").get_to(ndec);
    config.at("pa0").get_to(pa0);
    config.at("dpa").get_to(dpa);
    config.at("npoints_pa").get_to(npa);

    Scanning scan;
    Sky sky(nside_sky);
    PolBeam beam(nside_beam);
    Mapper mapper(nside_output_map);
    std::ofstream outputmap_file(output_map_path); 
    ConvolutionEngine conv(nside_sky, nside_beam, nra * ndec * npa);
    // buffers for detector data
    data_a = (float *)malloc(sizeof(float) * nra * ndec * npa);
    data_b = (float *)malloc(sizeof(float) * nra * ndec * npa);

    // make beams of detector a and b to be gaussian, no cross-pol
    beam.allocate_buffers();
    beam.load_beam_data_from_txt('a', det_a_beam_data);
    beam.load_beam_data_from_txt('b', det_b_beam_data);
    beam.normalize(nside_sky);

    // make sky to be a point source located at RA = 45deg, dec = 0.0 and 
    // be polarized in Q.
    sky.allocate_buffers();
    sky.make_point_source_sky(ra0 * DEG2RAD, dec0 * DEG2RAD, 1.0, 1.0, 0.0, 0.0);

    // make a raster scan around point source
    scan.make_raster_scan(nra, ndec, npa, 
        ra0 * DEG2RAD, dra * DEG2RAD, 
        dec0 * DEG2RAD, ddec * DEG2RAD, 
        pa0 * DEG2RAD, dpa * DEG2RAD);

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
    for(s = 0; s < scan.size(); s++)
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
    }
    auto end = std::chrono::steady_clock::now();
    time_wall = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // report
    std::cerr << "Time filling matrix: " << time_fill / 1000.0 << " ms" << std::endl;
    std::cerr << "Time transfering matrix: " << time_transfer / 1000.0 << " ms" << std::endl;
    std::cerr << "Time using matrix: " << time_conv / 1000.0 << " ms" << std::endl;
    std::cerr << "Time not accounted for: " << (time_fill + time_transfer + time_conv - time_wall) / 1000.0 << " ms" << std::endl;
    std::cerr << "[INFO] Computed " << nra * ndec * npa << " samples in " << time_wall / 1000.0 << " ms ";
    std::cerr << "(" << (nra * ndec * npa) / (time_wall / 1000000.0) << " samples/sec)" << std::endl;

    conv.data_to_host(data_a, data_b);

    // transform results to a map
    // detector b has a polarization angle of 90 degrees.
    mapper.accumulate_data(scan.size(), 
        scan.get_ra_ptr(), scan.get_dec_ptr(), scan.get_pa_ptr(), 0.0, data_a);
    //mapper.accumulate_data(scan.size(), 
    //    scan.get_ra_ptr(), scan.get_dec_ptr(), scan.get_pa_ptr(), 90.0 * DEG2RAD, data_b);
    mapper.solve_map();

    // output map to stdout
    float f = (1.0 * nside_sky) / (1.0 * nside_output_map);
    f = f * f;
    for(s = 0; s < mapper.size(); s++)
    {
        outputmap_file << mapper.get_stokes_I()[s] * f << " ";
        outputmap_file << mapper.get_stokes_Q()[s] * f << " ";
        outputmap_file << mapper.get_stokes_U()[s] * f << " ";
        outputmap_file << mapper.get_hitmap()[s];
        outputmap_file << std::endl;
    }
    outputmap_file.close();

    free(data_a);
    free(data_b);

    return 0;
}