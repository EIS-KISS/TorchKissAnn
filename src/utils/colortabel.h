#include <array>

namespace color
{

enum color: int
{
	BLACK = 0,
	RED,
	GREEN,
	BLUE,
	YELLOW,
	PINK,
	CYAN,
	GREY,
	DARK_RED,
	DARK_GREEN,
	DARK_BLUE,
	ORANGE,
	PURPLE,
	SKOBELOFF,
	WHITE
};

inline std::array<int, 3> getColor(const color index)
{
	switch(index)
	{
		case BLACK:
			return {0,0,0};
		case RED:
			return {255,0,0};
		case GREEN:
			return {0,255,0};
		case BLUE:
			return {0,0,255};
		case YELLOW:
			return {255,255,0};
		case PINK:
			return {255,0,255};
		case CYAN:
			return {0,255,255};
		case GREY:
			return {128,128,128};
		case DARK_RED:
			return {128,0,0};
		case DARK_GREEN:
			return {0,128,0};
		case DARK_BLUE:
			return {0,0,128};
		case ORANGE:
			return {128,128,0};
		case PURPLE:
			return {128,0,128};
		case SKOBELOFF:
			return {0,128,128};
		case WHITE:
		default:
			return {255,255,255};
	}
}

}
