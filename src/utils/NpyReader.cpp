#include "NpyReader.hpp"

NpyReader* NpyReader::instance = nullptr;

// Helper function to parse the header and extract metadata
void NpyReader::parseHeader(std::ifstream &file, std::vector<size_t> &shape, std::string &dtype)
{
    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
    {
        throw std::runtime_error("Not a valid .npy file");
    }

    // Read version number
    char version[2];
    file.read(version, 2);
    int major = version[0];
    int minor = version[1];

    // Read header length
    uint32_t header_len;
    if (major == 1)
    {
        uint16_t len16;
        file.read(reinterpret_cast<char *>(&len16), 2);
        header_len = len16;
    }
    else if (major == 2)
    {
        file.read(reinterpret_cast<char *>(&header_len), 4);
    }
    else
    {
        throw std::runtime_error("Unsupported .npy version");
    }

    // Read and parse the header
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);
    //std::cout << "Header: " << header << std::endl; // Debugging output

    // Extract shape string (without parentheses)
    size_t pos = header.find("'shape': (") + 9; // Find the start of shape
    if (pos == std::string::npos)
    {
        throw std::runtime_error("Shape not found in header");
    }

    size_t end_pos = header.find(")", pos);
    if (end_pos == std::string::npos)
    {
        throw std::runtime_error("Malformed shape in header");
    }

    // Extract the shape string and trim spaces
    std::string shape_str = header.substr(pos + 1, end_pos - pos);
    shape_str.erase(0, shape_str.find_first_not_of(" \t\n\r")); // Trim leading spaces
    shape_str.erase(shape_str.find_last_not_of(" \t\n\r") + 1); // Trim trailing spaces

    // Now split the shape string by commas and convert to size_t
    std::stringstream shape_stream(shape_str);
    std::string dim_str;
    while (std::getline(shape_stream, dim_str, ','))
    {
        try
        {
            dim_str.erase(0, dim_str.find_first_not_of(" \t\n\r")); // Trim each dimension string
            dim_str.erase(dim_str.find_last_not_of(" \t\n\r") + 1); // Trim trailing spaces
            if (dim_str != ")")
            {
                shape.push_back(std::stoul(dim_str)); // Convert the dimension string to size_t
            }
        }
        catch (const std::invalid_argument &e)
        {
            throw std::runtime_error("Invalid dimension in shape: " + dim_str);
        }
    }

    // Handle case where shape might only be one dimension (e.g., shape: (60000,))
    if (shape.size() == 1 && shape_str.back() == ',')
    {
        shape.push_back(1); // For single-dimensional arrays, the shape might have been like (60000,)
    }

    // Extract dtype
    size_t dtype_pos = header.find("'descr': '");
    if (dtype_pos == std::string::npos)
    {
        throw std::runtime_error("Could not find dtype in header");
    }
    dtype_pos += 10; // Move past "'descr': '"
    size_t dtype_end = header.find("'", dtype_pos);
    dtype = header.substr(dtype_pos, dtype_end - dtype_pos);
}

// Load .npy file with float32 data
std::vector<float> NpyReader::loadFloat(const std::string &filename, std::vector<size_t> &shape)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string dtype;
    parseHeader(file, shape, dtype);

    // Validate dtype (expecting float32)
    if (dtype != "<f4")
    {
        throw std::runtime_error("Unsupported data type (only float32 is supported)");
    }

    // Read array data
    size_t total_size = 1;
    for (size_t dim : shape)
    {
        total_size *= dim;
    }

    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char *>(data.data()), total_size * sizeof(float));

    return data;
}

// Load .npy file with uint8 data
std::vector<unsigned char> NpyReader::loadUint8(const std::string &filename, std::vector<size_t> &shape)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string dtype;
    parseHeader(file, shape, dtype);

    // Validate dtype (expecting uint8)
    if (dtype != "|u1")
    {
        throw std::runtime_error("Unsupported data type (only uint8 is supported)");
    }

    // Read array data
    size_t total_size = 1;
    for (size_t dim : shape)
    {
        total_size *= dim;
    }

    std::vector<unsigned char> data(total_size);
    file.read(reinterpret_cast<char *>(data.data()), total_size * sizeof(unsigned char));

    return data;
}