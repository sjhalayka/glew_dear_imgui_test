#ifndef LOGGING_SYSTEM_H
#define LOGGING_SYSTEM_H

#include <deque>
#include <string>

using std::deque;
using std::string;


class logging_system
{
protected:
	deque<string> contents;
	size_t max_size;

public:
	logging_system(void)
	{
		max_size = 10;
	}

	size_t get_max_size(void)
	{
		return max_size;
	}

	void set_max_size(size_t new_size)
	{
		max_size = new_size;
	}

	size_t get_contents_size(void)
	{
		return contents.size();
	}

	void get_string_from_contents(size_t index, string& s)
	{
		if (index >= max_size)
		{
			s = "";
		}
		else
		{
			s = contents[index];
		}
	}

	void add_string_to_contents(const string& s)
	{
		contents.push_back(s);

		if (contents.size() > max_size)
		{
			contents.pop_front();
		}
	}

};

#endif