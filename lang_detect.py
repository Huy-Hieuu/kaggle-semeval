"""Improved language detector for code snippets."""
import re

def detect_language(code):
    if not code or len(code) < 5:
        return "unknown"

    scores = {lang: 0 for lang in ['Python', 'Java', 'C++', 'C#', 'JavaScript', 'Go', 'C', 'PHP']}

    # ===== Python =====
    s = 0
    if re.search(r'\bdef\s+\w+\s*\(', code): s += 3
    if re.search(r'^import\s+\w+', code, re.M): s += 2
    if re.search(r'^from\s+\w+\s+import', code, re.M): s += 3
    if 'print(' in code: s += 2
    if re.search(r'\belif\b', code): s += 4
    if 'self.' in code: s += 2
    if '"""' in code or "'''" in code: s += 2
    if re.search(r'^\s+\S', code, re.M) and not re.search(r'[{};]\s*$', code, re.M): s += 2
    if 'raw_input(' in code or 'input()' in code: s += 2
    if re.search(r'\brange\s*\(', code): s += 2
    if re.search(r'\bfor\s+\w+\s+in\s+', code): s += 2
    if '__init__' in code or '__main__' in code: s += 3
    # Penalize if clearly not Python
    if '#include' in code: s -= 5
    if 'public class' in code or 'public static' in code: s -= 3
    scores['Python'] = max(s, 0)

    # ===== Java =====
    s = 0
    if re.search(r'\bpublic\s+(class|interface|enum)\s+\w+', code): s += 3
    if 'public static void main' in code: s += 4
    if 'System.out' in code: s += 3
    if re.search(r'import\s+java\.', code): s += 4
    if re.search(r'import\s+javax\.', code): s += 4
    if re.search(r'import\s+org\.', code): s += 2
    if '.println(' in code: s += 2
    if 'throws' in code and 'Exception' in code: s += 2
    if re.search(r'@Override|@SuppressWarnings|@Deprecated', code): s += 3
    if re.search(r'\bnew\s+\w+\s*\(', code): s += 1
    if re.search(r'(private|protected|public)\s+(static\s+)?(final\s+)?(void|int|String|boolean|long|double|float|List|Map|Set)\s+\w+', code): s += 2
    if 'extends ' in code or 'implements ' in code: s += 2
    # Java-specific: no $ (not PHP), no cout (not C++), no using System (not C#)
    if 'using System' in code: s -= 5
    if '$' in code and '<?php' in code: s -= 5
    scores['Java'] = max(s, 0)

    # ===== C++ =====
    s = 0
    if re.search(r'#include\s*<(iostream|vector|string|algorithm|map|set|queue|stack|cstring|cmath|cstdio|cstdlib|bits/stdc\+\+)', code): s += 4
    if 'cout' in code or 'cin' in code: s += 3
    if 'std::' in code: s += 3
    if 'using namespace std' in code: s += 4
    if 'nullptr' in code: s += 3
    if re.search(r'(vector|map|set|pair|string)<', code): s += 2
    if '::' in code and ('#include' in code or 'namespace' in code): s += 2
    if re.search(r'template\s*<', code): s += 3
    # Distinguish from C
    if 'cout' in code or 'cin' in code or 'std::' in code or 'class ' in code: s += 1
    scores['C++'] = max(s, 0)

    # ===== C# =====
    s = 0
    if 'using System' in code: s += 5
    if 'Console.Write' in code or 'Console.Read' in code: s += 4
    if re.search(r'\bnamespace\s+\w+', code) and 'std' not in code and 'package' not in code: s += 2
    if re.search(r'\bvar\s+\w+\s*=', code): s += 1
    if '.Length' in code and 'import java' not in code: s += 2
    if 'foreach' in code: s += 2
    if re.search(r'\basync\s+Task', code): s += 3
    if re.search(r'\bawait\s+', code) and 'using System' in code: s += 2
    if 'SafeHandle' in code or 'IntPtr' in code: s += 3
    if re.search(r'\[MethodImpl|^\s*\[.*\]\s*$', code, re.M): s += 1  # attributes
    if 'abstract class' in code and ('using System' in code or 'namespace' in code): s += 2
    if 'public override' in code or 'protected override' in code: s += 1
    if re.search(r'(string|int|bool|void|Task|List|Dictionary)\s+\w+', code) and 'using System' in code: s += 1
    # C# common patterns
    if 'IServiceProvider' in code or 'IHosting' in code or 'IEnumerable' in code: s += 3
    if re.search(r'<summary>|<param |<returns>', code): s += 2  # XML doc comments
    scores['C#'] = max(s, 0)

    # ===== JavaScript =====
    s = 0
    if re.search(r'\b(const|let|var)\s+\w+\s*=', code): s += 2
    if '=>' in code: s += 2
    if 'console.log' in code: s += 4
    if re.search(r'\brequire\s*\([\'"]', code): s += 3
    if 'module.exports' in code: s += 4
    if 'document.' in code or 'window.' in code: s += 3
    if 'process.stdin' in code or 'process.argv' in code: s += 3
    if re.search(r'function\s+\w+\s*\(', code) and '=>' in code: s += 1
    if 'class ' in code and 'extends' in code and 'constructor(' in code: s += 3
    if 'EventTarget' in code or 'addEventListener' in code: s += 3
    if 'Promise' in code or '.then(' in code: s += 2
    if 'export ' in code and ('default' in code or 'function' in code or 'class' in code): s += 2
    # Penalize if clearly not JS
    if '#include' in code or 'import java' in code or 'using System' in code: s -= 5
    if '<?php' in code: s -= 5
    scores['JavaScript'] = max(s, 0)

    # ===== Go =====
    s = 0
    if re.search(r'^package\s+\w+', code, re.M): s += 4
    if re.search(r'\bfunc\s+(\([\w\s*]+\)\s*)?\w+\s*\(', code): s += 3
    if 'fmt.' in code: s += 3
    if ':=' in code: s += 2
    if re.search(r'import\s*\(', code): s += 3
    if re.search(r'import\s+"', code): s += 3
    if 'func main()' in code: s += 2
    if re.search(r'\bgo\s+\w+', code): s += 2
    if 'chan ' in code or '<-' in code: s += 2
    if re.search(r'\bdefer\s+', code): s += 3
    if 'interface {' in code or 'struct {' in code: s += 2
    if re.search(r'\bmap\[', code): s += 2
    scores['Go'] = max(s, 0)

    # ===== C =====
    s = 0
    if '#include <stdio.h>' in code or '#include <stdlib.h>' in code: s += 4
    if '#include <string.h>' in code or '#include <math.h>' in code: s += 3
    if 'printf(' in code: s += 3
    if 'scanf(' in code: s += 3
    if 'malloc(' in code or 'calloc(' in code or 'free(' in code: s += 3
    if re.search(r'int\s+main\s*\(', code): s += 1
    if re.search(r'#include\s*"[\w.]+\.h"', code): s += 2  # local .h includes
    if 'typedef ' in code and 'struct' in code: s += 2
    if re.search(r'\bstruct\s+\w+\s*\{', code) and 'class' not in code: s += 1
    if 'void *' in code or 'char *' in code or 'int *' in code: s += 1
    if 'NULL' in code and '#include' in code: s += 1
    # Penalize if C++ indicators
    if 'cout' in code or 'cin' in code or 'std::' in code: s -= 10
    if '#include <iostream>' in code or '#include <vector>' in code: s -= 10
    if 'class ' in code and '::' in code: s -= 5
    if 'template<' in code or 'template <' in code: s -= 10
    # Penalize if clearly other languages
    if 'using System' in code: s -= 10
    if 'import java' in code: s -= 10
    scores['C'] = max(s, 0)

    # ===== PHP =====
    s = 0
    if '<?php' in code: s += 6
    if re.search(r'\$\w+\s*=', code): s += 2
    if re.search(r'\$\w+', code) and '->' in code: s += 2
    if 'echo ' in code: s += 2
    if re.search(r'function\s+\w+\s*\(.*\$', code): s += 3
    if 'array(' in code or '[]' in code and '$' in code: s += 1
    # Penalize if $ is in other context (bash, etc)
    if '#include' in code or 'import java' in code: s -= 5
    scores['PHP'] = max(s, 0)

    # ===== Fallback heuristics for remaining unknowns =====
    # C-family detection (C or C++) by structure
    if scores['C'] < 3 and scores['C++'] < 3:
        if '#include' in code:
            if re.search(r'(cout|cin|std::|class\s+\w+|vector<|string>|namespace)', code):
                scores['C++'] += 4
            elif re.search(r'(printf|scanf|malloc|free|NULL|void\s*\*)', code):
                scores['C'] += 4
            else:
                # Generic #include — lean C if no OOP signs
                if 'class ' not in code and '::' not in code:
                    scores['C'] += 3
                else:
                    scores['C++'] += 3

    # Java fallback: lots of public/private methods with types
    if scores['Java'] < 3:
        java_methods = len(re.findall(r'(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\(', code))
        if java_methods >= 2 and 'using System' not in code and '$' not in code:
            # Could be Java or C# — check for Java-specific
            if '.get(' in code or '.set(' in code or '.add(' in code or 'List<' in code:
                scores['Java'] += 3

    # C# fallback: public methods + namespace but not Java indicators
    if scores['C#'] < 3:
        if re.search(r'(public|private|protected)\s+(override\s+|virtual\s+|abstract\s+)', code):
            if 'import java' not in code and not re.search(r'@\w+', code):
                scores['C#'] += 3
        if re.search(r'\bnew\s+\w+\s*\[', code) and '.Length' in code:
            scores['C#'] += 2

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]

    if best_score >= 3:
        return best_lang

    # Last resort heuristics
    if '{' in code and ';' in code:
        # C-family syntax
        if '#include' in code:
            return 'C'
        if re.search(r'(public|private|protected)\s+', code):
            if '$' in code:
                return 'PHP'
            if 'import ' in code:
                return 'Java'
            return 'Java'  # most common in training

    return "unknown"
