// (Note: this part is not required for understanding how Genifer works)

// Train Genifer to convert Chinese texts to Cantonese texts
// ==========================================================
// Cantonese is a dialect of Chinese very close to standard Chinese, so only minor
// transliterations are required to turn Chinese into Cantonese.

// 问题是如何 define states and actions。  从 agent 个体的角度看，当然是字／词的 input-output
// = states，还有将 focus 在上下文移动。   还有对 reasoning operator 的控制。
// 那就是 RNN 对不同输入有不同输出，输出正确或错误时都应该有 learning。
// 当 first impression 失败时就用 forward-backward。  The training errors will have some patterns,
// as well as the local gradients.

// TO-DO:
// * structure of K is fixed: K = in-word, in-strength, speak-strength, internal, out-word
// * need vector representation for individual words

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <wchar.h>

// List of Chinese characters that are recognized by the system
// Chinese characters not in these lists are considered "don't care"
wchar_t inChars[1024] = L"";
wchar_t outChars[1024] = L"";
// index to the above arrays
int inCharsNum = 0;
int outCharsNum = 0;

// ******************* read training data and testing data **********************

void addInChars(wchar_t *newStr)
	{
	for (int i = 0; i < wcslen(newStr); ++i)
		{
		wchar_t c = newStr[i];
		if (wcschr(inChars, c) == NULL)
			{
			inChars[inCharsNum] = c;
			inCharsNum++;
			inChars[inCharsNum] = '\0';
			}
		}
	}

void addOutChars(wchar_t *newStr)
	{
	for (int i = 0; i < wcslen(newStr); ++i)
		{
		wchar_t c = newStr[i];
		if (wcschr(outChars, c) == NULL)
			{
			outChars[outCharsNum] = c;
			outCharsNum++;
			outChars[outCharsNum] = '\0';
			}
		}
	}

void find_unique_chars()
	{
	wchar_t s[1024];

	wprintf(L"Finding unique characters...\n");

	FILE *fp1;
	if((fp1 = fopen("/home/yky/NetBeansProjects/conceptual-keyboard/training-set.txt", "r")) == NULL)
		{ printf("cannot open training-set.txt\n"); exit(1); }

	while (fwscanf(fp1, L"%S", s) > 0) {
		wprintf(s);
		wprintf(L"\n");
		addInChars(s);

		fwscanf(fp1, L"%S", s);
		wprintf(s);
		wprintf(L"\n");
		addOutChars(s);
		}

	fclose(fp1);

	wprintf(inChars);
	wprintf(L"\n");
	wprintf(outChars);
	wprintf(L"\n");
	}

//**************************main function***********************//

int test_Chinese()
	{
    setlocale(LC_ALL, "");
	wprintf(L"*** 欢迎使用珍妮花 5.3 ***\n\n");		// "Welcome to Genifer 5.3"

	find_unique_chars();

	/*output data to a file
	FILE *fout;
	if ((fout = fopen("randomtest_1.txt", "w")) == NULL)
		{ fprintf(stderr, "file open failed.\n"); exit(1); }

	fclose(fout);
	*/

	return 0;
	}
