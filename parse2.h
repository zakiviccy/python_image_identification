#ifndef __PARSE_H__
#define __PARSE_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* 感染データを格納する構造体の定義 */
struct infection{
  int year;
  int month; 
  int day; 
  char prefecture[20];
  int infected;
  int hospital;
  int discharge;
  int death;
};

/* 宣言 */
void str2infection(char str[], struct infection *inf);
void print_infection(struct infection *inf);
void print_save_infection(struct infection *inf, FILE *fp);

#endif
