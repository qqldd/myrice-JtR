/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 1996-98,2010,2012 by Solar Designer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "arch.h"
#include "misc.h"
#include "memory.h"
#include "common.h"
#include "johnswap.h"

unsigned int mem_saving_level = 0;

// Add 'cleanup' methods for the mem_tiny_alloc.  VERY little cost, but
// allows us to check for mem leaks easier.
struct rm_list
{
	void *mem;
	struct rm_list *next;
};
static struct rm_list *mem_alloc_tiny_memory;

static void add_memory_link(void *v) {
	struct rm_list *p = mem_alloc(sizeof(struct rm_list));
	p->next = mem_alloc_tiny_memory;
	p->mem = v;
	mem_alloc_tiny_memory = p;
}
// call at program exit.
void cleanup_tiny_memory()
{
	struct rm_list *p = mem_alloc_tiny_memory, *p2;
	for (;;) {
		if (!p)
			return;
		free(p->mem);
		p2 = p->next;
		free(p);
		p = p2;
	}
}

void *mem_alloc(size_t size)
{
	void *res;

	if (!size) return NULL;

	if (!(res = malloc(size))) {
		fprintf(stderr, "malloc: %s\n", strerror(ENOMEM));
		error();
	}

	return res;
}

void *mem_alloc_tiny(size_t size, size_t align)
{
	static char *buffer = NULL;
	static size_t bufree = 0;
	size_t mask;
	char *p;

#if ARCH_ALLOWS_UNALIGNED
	if (mem_saving_level > 2)
		align = MEM_ALIGN_NONE;
#endif

	mask = align - 1;

	do {
		if (buffer) {
			size_t need =
			    size + mask - (((size_t)buffer + mask) & mask);
			if (bufree >= need) {
				p = buffer;
				p += mask;
				p -= (size_t)p & mask;
				bufree -= need;
				buffer = p + size;
				return p;
			}
		}

		if (size + mask > MEM_ALLOC_SIZE ||
		    bufree > MEM_ALLOC_MAX_WASTE)
			break;

		buffer = mem_alloc(MEM_ALLOC_SIZE);
		add_memory_link((void*)buffer);
		bufree = MEM_ALLOC_SIZE;
	} while (1);

	p = mem_alloc(size + mask);
	p += mask;
	p -= (size_t)p & mask;
	return p;
}

void *mem_calloc_tiny(size_t size, size_t align) {
	char *cp = (char*) mem_alloc_tiny(size, align);
	memset(cp, 0, size);
	return cp;
}

void *mem_alloc_copy(void *src, size_t size, size_t align)
{
	return memcpy(mem_alloc_tiny(size, align), src, size);
}

char *str_alloc_copy(char *src)
{
	size_t size;

	if (!src) return "";
	if (!*src) return "";

	size = strlen(src) + 1;
	return (char *)memcpy(mem_alloc_tiny(size, MEM_ALIGN_NONE), src, size);
}

void dump_stuff(void* x, unsigned int size)
{
        unsigned int i;
        for(i=0;i<size;i++)
        {
	        printf("%.2x", ((unsigned char*)x)[i]);
                if( (i%4)==3 )
                        printf(" ");
        }
        printf("\n");
}
void dump_stuff_msg(void *msg, void *x, unsigned int size) {
	printf("%s : ", (char *)msg);
	dump_stuff(x, size);
}

#if defined(MMX_COEF) || defined(NT_X86_64) || defined (MD5_SSE_PARA) || defined (MD4_SSE_PARA) || defined (SHA1_SSE_PARA)
#ifndef MMX_COEF
#define MMX_COEF	4
#endif

#if ARCH_ALLOWS_UNALIGNED
void alter_endianity(void * _x, unsigned int size)
{
	// size is in BYTES
	// since we are only using this in MMX code, we KNOW that we are using x86 CPU's which do not have problems
	// with non aligned 4 byte word access.  Thus, we use a faster swapping function.
	ARCH_WORD_32 *x = (ARCH_WORD_32*)_x;
	int i = -1;
	size>>=2;
	while (++i < size) {
		x[i] = JOHNSWAP(x[i]);
	}
}
#endif

// These work for standard MMX_COEF buffers, AND for SSEi MMX_PARA multiple MMX_COEF blocks, where index will be mod(X * MMX_COEF) and not simply mod(MMX_COEF)
#define SHAGETPOS(i, index)		( (index&(MMX_COEF-1))*4 + ((i)&(0xffffffff-3) )*MMX_COEF + (3-((i)&3)) + (index>>(MMX_COEF>>1))*SHA_BUF_SIZ*4*MMX_COEF ) //for endianity conversion
#define SHAGETOUTPOS(i, index)		( (index&(MMX_COEF-1))*4 + ((i)&(0xffffffff-3) )*MMX_COEF + (3-((i)&3)) + (index>>(MMX_COEF>>1))*20*MMX_COEF ) //for endianity conversion
#define GETPOS(i, index)		( (index&(MMX_COEF-1))*4 + ((i)&(0xffffffff-3) )*MMX_COEF +    ((i)&3)  + (index>>(MMX_COEF>>1))*64*MMX_COEF  )
#define GETOUTPOS(i, index)		( (index&(MMX_COEF-1))*4 + ((i)&(0xffffffff-3) )*MMX_COEF +    ((i)&3)  + (index>>(MMX_COEF>>1))*16*MMX_COEF  )

void dump_stuff_mmx(void *buf, unsigned int size, unsigned int index)
{
	unsigned int i;
	for(i=0;i<size;i++)
	{
		printf("%.2x", ((unsigned char*)buf)[GETPOS(i, index)]);
		if( (i%4)==3 )
			printf(" ");
	}
	printf("\n");
}
void dump_out_mmx(void *buf, unsigned int size, unsigned int index)
{
	unsigned int i;
	for(i=0;i<size;i++)
	{
		printf("%.2x", ((unsigned char*)buf)[GETOUTPOS(i, index)]);
		if( (i%4)==3 )
			printf(" ");
	}
	printf("\n");
}
void dump_stuff_mmx_msg(void *msg, void *buf, unsigned int size, unsigned int index) {
	printf("%s : ", (char*)msg);
	dump_stuff_mmx(buf, size, index);
}
void dump_out_mmx_msg(void *msg, void *buf, unsigned int size, unsigned int index) {
	printf("%s : ", (char*)msg);
	dump_out_mmx(buf, size, index);
}

void dump_stuff_shammx(void *buf, unsigned int size, unsigned int index)
{
	unsigned int i;
	for(i=0;i<size;i++)
	{
		printf("%.2x", ((unsigned char*)buf)[SHAGETPOS(i, index)]);
		if( (i%4)==3 )
			printf(" ");
	}
	printf("\n");
}
void dump_stuff_shammx_msg(void *msg, void *buf, unsigned int size, unsigned int index) {
	printf("%s : ", (char*)msg);
	dump_stuff_shammx(buf, size, index);
}
void dump_out_shammx(void *buf, unsigned int size, unsigned int index)
{
	unsigned int i;
	for(i=0;i<size;i++)
	{
		printf("%.2x", ((unsigned char*)buf)[SHAGETOUTPOS(i, index)]);
		if( (i%4)==3 )
			printf(" ");
	}
	printf("\n");
}
void dump_out_shammx_msg(void *msg, void *buf, unsigned int size, unsigned int index) {
	printf("%s : ", (char*)msg);
	dump_out_shammx(buf, size, index);
}

#endif

void alter_endianity_w(void *_x, unsigned int count) {
	int i = -1;
	ARCH_WORD_32 *x = (ARCH_WORD_32*)_x;
#if ARCH_ALLOWS_UNALIGNED
	while (++i < count) {
		x[i] = JOHNSWAP(x[i]);
	}
#else
	unsigned char *cpX, c;
	if (is_aligned(x,sizeof(ARCH_WORD_32))) {
		// we are in alignment.
		while (++i < count) {
			x[i] = JOHNSWAP(x[i]);
		}
		return;
	}
	// non-aligned data :(
	cpX = (unsigned char*)x;
	while (++i < count) {
		c = *cpX;
		*cpX = cpX[3];
		cpX[3] = c;
		c = cpX[1];
		cpX[1] = cpX[2];
		cpX[2] = c;
		cpX += 4;
	}
#endif
}
