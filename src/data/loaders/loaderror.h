#include <exception>

class loader_error: public std::exception
{
	std::string whatStr;
public:
	loader_error(const std::string& whatIn): whatStr(whatIn)
	{}
	virtual const char* what() const noexcept override
	{
		return whatStr.c_str();
	}
};
