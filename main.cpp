#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "dear imgui/imgui.h"
#include "dear imgui/imgui_impl_glfw.h"
#include "dear imgui/imgui_impl_opengl3.h"


// Automatically link in the GLFW and GLEW libraries if compiling on MSVC++
#ifdef _MSC_VER
#pragma comment(lib, "glew32")
#pragma comment(lib, "glfw3") // https://github.com/glfw/glfw/releases/download/3.3.2/glfw-3.3.2.bin.WIN64.zip
#endif


#include <iostream>
using namespace std;

int main(void)
{
	if (false == glewInit())
		return 1;



	return 0;
}