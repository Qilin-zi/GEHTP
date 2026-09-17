#pragma once
#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <stdexcept>

namespace hnnx {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;

    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        for (const auto& [k, v] : obj_val) if (k == key) return true;
        return false;
    }

    const JsonValue& at(const std::string& key) const {
        if (!is_object()) throw std::runtime_error("not an object");
        for (const auto& [k, v] : obj_val) if (k == key) return v;
        throw std::runtime_error("key not found: " + key);
    }

    const JsonValue& at(size_t idx) const {
        if (!is_array()) throw std::runtime_error("not an array");
        if (idx >= arr_val.size()) throw std::runtime_error("index out of range");
        return arr_val[idx];
    }

    size_t size() const {
        if (is_array()) return arr_val.size();
        if (is_object()) return obj_val.size();
        return 0;
    }

    int64_t as_int() const { return static_cast<int64_t>(num_val); }
    double as_num() const { return num_val; }
    const std::string& as_str() const { return str_val; }
    bool as_bool() const { return bool_val; }
};

inline JsonValue parse_json(const std::string& text);

namespace json_detail {

struct Parser {
    const char* p;
    const char* end;

    Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void skip_ws() {
        while (p < end) {
            char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { p++; }
            else break;
        }
    }

    char peek() {
        if (p >= end) throw std::runtime_error("unexpected EOF");
        return *p;
    }

    char next() {
        if (p >= end) throw std::runtime_error("unexpected EOF");
        return *p++;
    }

    void expect(char c) {
        skip_ws();
        if (next() != c) throw std::runtime_error(std::string("expected '") + c + "'");
    }

    JsonValue parseValue() {
        skip_ws();
        if (p >= end) throw std::runtime_error("unexpected EOF");
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        throw std::runtime_error(std::string("unexpected char: ") + c);
    }

    JsonValue parseString() {
        JsonValue v;
        v.type = JsonValue::Type::String;
        expect('"');
        while (p < end) {
            char c = *p++;
            if (c == '"') return v;
            if (c == '\\') {
                if (p >= end) throw std::runtime_error("bad escape");
                char e = *p++;
                switch (e) {
                    case '"': v.str_val += '"'; break;
                    case '\\': v.str_val += '\\'; break;
                    case '/': v.str_val += '/'; break;
                    case 'n': v.str_val += '\n'; break;
                    case 't': v.str_val += '\t'; break;
                    case 'r': v.str_val += '\r'; break;
                    case 'b': v.str_val += '\b'; break;
                    case 'f': v.str_val += '\f'; break;
                    case 'u': {
                        if (p + 4 > end) throw std::runtime_error("bad unicode");
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; i++) {
                            char h = *p++;
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else throw std::runtime_error("bad hex");
                        }
                        if (cp < 0x80) v.str_val += static_cast<char>(cp);
                        else if (cp < 0x800) {
                            v.str_val += static_cast<char>(0xC0 | (cp >> 6));
                            v.str_val += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            v.str_val += static_cast<char>(0xE0 | (cp >> 12));
                            v.str_val += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            v.str_val += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: v.str_val += e; break;
                }
            } else {
                v.str_val += c;
            }
        }
        throw std::runtime_error("unterminated string");
    }

    JsonValue parseNumber() {
        JsonValue v;
        v.type = JsonValue::Type::Number;
        const char* start = p;
        if (peek() == '-') p++;
        while (p < end && (*p >= '0' && *p <= '9')) p++;
        bool is_float = false;
        if (p < end && *p == '.') { is_float = true; p++; while (p < end && (*p >= '0' && *p <= '9')) p++; }
        if (p < end && (*p == 'e' || *p == 'E')) { is_float = true; p++; if (p < end && (*p == '+' || *p == '-')) p++; while (p < end && (*p >= '0' && *p <= '9')) p++; }
        std::string num_str(start, p - start);
        v.num_val = std::stod(num_str);
        return v;
    }

    JsonValue parseBool() {
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        if (p + 4 <= end && p[0] == 't' && p[1] == 'r' && p[2] == 'u' && p[3] == 'e') {
            v.bool_val = true; p += 4;
        } else if (p + 5 <= end && p[0] == 'f' && p[1] == 'a' && p[2] == 'l' && p[3] == 's' && p[4] == 'e') {
            v.bool_val = false; p += 5;
        } else {
            throw std::runtime_error("bad bool literal");
        }
        return v;
    }

    JsonValue parseNull() {
        JsonValue v;
        v.type = JsonValue::Type::Null;
        if (p + 4 <= end && p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l') {
            p += 4;
        } else {
            throw std::runtime_error("bad null literal");
        }
        return v;
    }

    JsonValue parseArray() {
        JsonValue v;
        v.type = JsonValue::Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') { p++; return v; }
        for (;;) {
            v.arr_val.push_back(parseValue());
            skip_ws();
            char c = next();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("expected ',' or ']'");
        }
        return v;
    }

    JsonValue parseObject() {
        JsonValue v;
        v.type = JsonValue::Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') { p++; return v; }
        for (;;) {
            skip_ws();
            JsonValue key = parseString();
            skip_ws();
            expect(':');
            JsonValue val = parseValue();
            v.obj_val.emplace_back(key.str_val, std::move(val));
            skip_ws();
            char c = next();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("expected ',' or '}'");
        }
        return v;
    }
};

}

inline JsonValue parse_json(const std::string& text) {
    json_detail::Parser parser(text);
    JsonValue result = parser.parseValue();
    parser.skip_ws();
    return result;
}

inline JsonValue parse_json_file(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open: " + path);
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string text(sz, '\0');
    std::fread(text.data(), 1, sz, fp);
    std::fclose(fp);
    return parse_json(text);
}

} // namespace hnnx
