#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <sstream>

class NpyReader {
private:
    // Private static instance for Singleton
    static NpyReader* instance;

    // Private constructor for Singleton
    NpyReader() {}

    // Helper function to parse the header and extract metadata
    void parseHeader(std::ifstream& file, std::vector<size_t>& shape, std::string& dtype);

public:
    // Delete copy constructor and assignment operator for Singleton
    NpyReader(const NpyReader&) = delete;
    NpyReader& operator=(const NpyReader&) = delete;

    // Static function to get the singleton instance
    static NpyReader* getInstance() {
        if (!instance) {
            instance = new NpyReader();
        }
        return instance;
    }

    // Load .npy file with float32 data
    std::vector<float> loadFloat(const std::string& filename, std::vector<size_t>& shape);

    // Load .npy file with uint8 data
    std::vector<unsigned char> loadUint8(const std::string& filename, std::vector<size_t>& shape);
};

