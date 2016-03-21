#ifndef LOG
#define LOG

class Silent
{
    template <typename... Params>
    static void init(Params... params) {}

    template <typename... Params>
    static void step(Params... params) {}

    template <typename... Params>
    static void finalize(Params... params) {}
}

class StandardLog
{
    template <typename... Params>
    static void init(Params... params) {}

    template <typename... Params>
    static void step(Params... params) {}

    template <typename... Params>
    static void finalize(Params... params) {}
}


#endif // LOG
