#ifndef BMP_H
#define BMP_H

#include <atomic>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <string>
#include <iostream>
#include <chrono>
#include <sstream>
using namespace std;

class BMP
{
public:
	std::uint32_t width, height;
	std::uint16_t BitsPerPixel;
	std::vector<unsigned char> Pixels;

public:
	BMP(void)
	{

	}


	bool load(const char* FilePath)
	{
		std::fstream hFile(FilePath, std::ios::in | std::ios::binary);
		if (!hFile.is_open()) return false;

		hFile.seekg(0, std::ios::end);
		std::size_t Length = hFile.tellg();
		hFile.seekg(0, std::ios::beg);
		std::vector<std::uint8_t> FileInfo(Length);
		hFile.read(reinterpret_cast<char*>(FileInfo.data()), 54);

		if (FileInfo[0] != 'B' && FileInfo[1] != 'M')
		{
			hFile.close();
			return false;
		}

		// if (FileInfo[28] != 24 && FileInfo[28] != 32)
		if (FileInfo[28] != 32)
		{
			hFile.close();
			return false;
		}

		BitsPerPixel = FileInfo[28];
		width = FileInfo[18] + (FileInfo[19] << 8);
		height = FileInfo[22] + (FileInfo[23] << 8);
		std::uint32_t PixelsOffset = FileInfo[10] + (FileInfo[11] << 8);
		std::uint32_t size = ((width * BitsPerPixel + 31) / 32) * 4 * height;
		Pixels.resize(size);

		hFile.seekg(PixelsOffset, std::ios::beg);
		hFile.read(reinterpret_cast<char*>(Pixels.data()), size);
		hFile.close();


		// Reverse row order
		short unsigned int num_rows_to_swap = height;
		vector<unsigned char> buffer(static_cast<size_t>(width) * 4);

		if (0 != height % 2)
			num_rows_to_swap--;

		num_rows_to_swap /= 2;

		for (size_t i = 0; i < num_rows_to_swap; i++)
		{
			size_t y_first = i * static_cast<size_t>(width) * 4;
			size_t y_last = (static_cast<size_t>(height) - 1 - i) * static_cast<size_t>(width) * 4;

			memcpy(&buffer[0], &Pixels[y_first], static_cast<size_t>(width) * 4);
			memcpy(&Pixels[y_first], &Pixels[y_last], static_cast<size_t>(width) * 4);
			memcpy(&Pixels[y_last], &buffer[0], static_cast<size_t>(width) * 4);
		}

		return true;
	}


	std::vector<std::uint8_t> GetPixels() const { return this->Pixels; }
	std::uint32_t GetWidth() const { return this->width; }
	std::uint32_t GetHeight() const { return this->height; }
	bool HasAlphaChannel() { return BitsPerPixel == 32; }
};

#endif