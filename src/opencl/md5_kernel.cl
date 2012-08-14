/* MD5 OpenCL kernel based on Solar Designer's MD5 algorithm implementation at:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * This software is Copyright © 2010, Dhiru Kholia <dhiru.kholia at gmail.com>,
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted.
 *
 * Useful References:
 * 1. CUDA MD5 Hashing Experiments, http://majuric.org/software/cudamd5/
 * 2. oclcrack, http://sghctoma.extra.hu/index.php?p=entry&id=11
 * 3. http://people.eku.edu/styere/Encrypt/JS-MD5.html
 * 4. http://en.wikipedia.org/wiki/MD5#Algorithm */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : disable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

/* Macros for reading/writing chars from int32's (from rar_kernel.cl) */
#define GETCHAR(buf, index) (((uchar*)(buf))[(index)])
#define PUTCHAR(buf, index, val) (buf)[(index)>>2] = ((buf)[(index)>>2] & ~(0xffU << (((index) & 3) << 3))) + ((val) << (((index) & 3) << 3))

/* The basic MD5 functions */
#define Fc(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
#define F(x, y, z)			bitselect((z), (y), (x))
#define G(x, y, z)			bitselect((y), (x), (z))
//#define F(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
//#define G(x, y, z)			((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)			((x) ^ (y) ^ (z))
#define I(x, y, z)			((y) ^ ((x) | ~(z)))

/* The MD5 transformation for all four rounds. */
#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = rotate(a, (uint)s); \
    (a) += (b);

#define GET(i) (key[(i)])

__constant char alpha_set[] = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
};
#define ALPHA_SET_SIZE 52

#define MD5_HASH_SHR 2
#define MD5_PASSWORD_HASH_SIZE_0 0x10000
#define MD5_PASSWORD_HASH_SIZE_1 0x10000 // 64K
#define MD5_PASSWORD_HASH_SIZE_2 0x1000000 // 16M

static uint get_hash_low(uint hash) { return hash & 0xFFFF; }
static uint get_hash_high(uint hash) { return hash & 0xFFFFFF; }

// kernel void create_bitmaps(global const uint *data_info, global const uint* loaded_hash, global uint* bitmaps, global int* hashtable, global int* loaded_next_hash, global int* semaphor)
// {
//     const uint loaded_count = data_info[0];
//     const uint hash_num = data_info[1];
//     const uint id = get_global_id(0);

//     if (id < loaded_count) {
//         uint hash;
//         uint l_hash = loaded_hash[id];
        
//         if (hash_num == MD5_PASSWORD_HASH_SIZE_2)
//             hash = get_hash_high(l_hash);
//         else
//             hash = get_hash_low(l_hash);
//         uint index = hash / (sizeof(*bitmaps) * 8);
//         uint bit_index = hash % (sizeof(*bitmaps) * 8);
//         uint val = 1U << bit_index;
        
//         GetSemaphor(semaphor);
//         atom_xchg(&bitmaps[index], bitmaps[index] | val);
//         hash >>= MD5_HASH_SHR;
//         int h = hashtable[hash];
//         atom_xchg(&loaded_next_hash[id], h);
//         atom_xchg(&hashtable[hash], id);
//         ReleaseSemaphor(semaphor);
//     }
// }

/* some constants used below magically appear after make */
//#define KEY_LENGTH (MD5_PLAINTEXT_LENGTH + 1)

inline int bitmaps_val(uint hash, bool use_local, global uint* bitmaps[4], local uint* local_bitmaps, int index, uint bitmaps_num)
{
    int val = 0;
    if (use_local) 
        hash = get_hash_low(hash);
    else 
        hash = get_hash_high(hash);

    if (use_local)
        val = local_bitmaps[index * bitmaps_num + hash / (sizeof(*bitmaps[0]) *8)] & (1U << (hash % (sizeof(*bitmaps[0]) *8)));
    else
        val =  bitmaps[index][hash / (sizeof(*bitmaps[0]) *8)] &
            (1U << (hash % (sizeof(*bitmaps[0]) *8)));
    return val;
    
}
/* OpenCL kernel entry point. Copy KEY_LENGTH bytes key to be hashed from
 * global to local (thread) memory. Break the key into 16 32-bit (uint)
 * words. MD5 hash of a key is 128 bit (uint4). */
__kernel void md5(__global uint *data_info, __global const uint * keys, __global uint * hashes, __global uint* loaded_hash,  __global uint *matched_count, __global uint* cracked_count, __global uint* matched_keys, global uint* bitmaps0, global uint* bitmaps1, global uint* bitmaps2, global uint* bitmaps3, global int* hashtable, global int* loaded_next_hash, local uint* local_bitmaps)
{
	int id = get_global_id(0);
    int lws = get_local_size(0);
    int lid = get_local_id(0);
    int init_count = *cracked_count;
    init_count = (init_count + lws -1) / lws * lws;
    
    if (id < init_count) {
        uint num_keys = data_info[1];
        uint KEY_LENGTH = data_info[0] + 1;
        uint loaded_count = data_info[2];
        int base = id * (KEY_LENGTH / 4);
        uint hash_num = data_info[3];
        global uint* bitmaps[4] = {bitmaps0, bitmaps1, bitmaps2, bitmaps3};

#if 0
        __private uint p_loaded_hash[2];
#endif
        bool use_local = false;
        int bitmaps_num;

#if 0
        if (loaded_count < 3) {
            for (int i = 0; i < loaded_count; ++i)
                p_loaded_hash[i] = loaded_hash[i];
        } else
#endif
	if (hash_num == MD5_PASSWORD_HASH_SIZE_1) {
            bitmaps_num = (hash_num+sizeof(int)*8-1)/(sizeof(int)*8);

            for (int i = 0; i < bitmaps_num; i+=lws) {
                uint index = i+lid;
                if (index < bitmaps_num) {
                    local_bitmaps[index] = bitmaps[0][index];
                    local_bitmaps[index+bitmaps_num] = bitmaps[1][index];
                    local_bitmaps[index+bitmaps_num*2] = bitmaps[2][index];
                    local_bitmaps[index+bitmaps_num*3] = bitmaps[3][index];                    
                }
            }
            use_local = true;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        uint key[16] = { 0 };
        uint i;

        for (i = 0; i != (KEY_LENGTH / 4) && keys[base + i]; i++)
            key[i] = keys[base + i];

        /* padding code (borrowed from MD5_eq.c) */
        char *p = (char *) key;
        for (i = 0; i != 64 && p[i]; i++);

        int origin_i = i;
        int loop_num = loaded_count == 0 ? 0 : ALPHA_SET_SIZE;
        // -1 for add none character
        for (int alpha_i = -1; alpha_i < loop_num; ++alpha_i) {
            for (int alpha_j = -1; alpha_j < loop_num; ++alpha_j) {

                i = origin_i;
                // Generate key
                if (alpha_i != -1 && alpha_j != -1) {
                    PUTCHAR(key, i, alpha_set[alpha_i]);
                    ++i;
                    PUTCHAR(key, i, alpha_set[alpha_j]); 
                    ++i;                    
                } else if ((alpha_i != -1 && alpha_j == -1) || (alpha_i == -1 && alpha_j != -1))
                    continue;
                uint length = i;
                
                //p[i] = 0x80;
                //p[56] = i << 3;
                //p[57] = i >> 5;

                PUTCHAR(key, i, 0x80);
                PUTCHAR(key, 56, i << 3);
                PUTCHAR(key, 57, i >> 5);

                uint a, b, c, d;
                a = 0x67452301;
                b = 0xefcdab89;
                c = 0x98badcfe;
                d = 0x10325476;

                /* Round 1 */
/* We use Fc() instead of F() to let the compiler compute constant
 * subexpressions, which it apparently fails to do when we use bitselect().
 * It'd be better to do such precomputation manually, like it's done in
 * phpass_kernel.cl: phpass(). */
                STEP(Fc, a, b, c, d, GET(0), 0xd76aa478, 7);
                STEP(Fc, d, a, b, c, GET(1), 0xe8c7b756, 12);
                STEP(Fc, c, d, a, b, GET(2), 0x242070db, 17);
                STEP(Fc, b, c, d, a, GET(3), 0xc1bdceee, 22);
                STEP(F, a, b, c, d, GET(4), 0xf57c0faf, 7);
                STEP(F, d, a, b, c, GET(5), 0x4787c62a, 12);
                STEP(F, c, d, a, b, GET(6), 0xa8304613, 17);
                STEP(F, b, c, d, a, GET(7), 0xfd469501, 22);
                STEP(F, a, b, c, d, GET(8), 0x698098d8, 7);
                STEP(F, d, a, b, c, GET(9), 0x8b44f7af, 12);
                STEP(F, c, d, a, b, GET(10), 0xffff5bb1, 17);
                STEP(F, b, c, d, a, GET(11), 0x895cd7be, 22);
                STEP(F, a, b, c, d, GET(12), 0x6b901122, 7);
                STEP(F, d, a, b, c, GET(13), 0xfd987193, 12);
                STEP(F, c, d, a, b, GET(14), 0xa679438e, 17);
                STEP(F, b, c, d, a, 0, 0x49b40821, 22);

                /* Round 2 */
                STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5);
                STEP(G, d, a, b, c, GET(6), 0xc040b340, 9);
                STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14);
                STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20);
                STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5);
                STEP(G, d, a, b, c, GET(10), 0x02441453, 9);
                STEP(G, c, d, a, b, 0, 0xd8a1e681, 14);
                STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20);
                STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5);
                STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9);
                STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14);
                STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20);
                STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5);
                STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9);
                STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14);
                STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20);

                /* Round 3 */
                STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4);
                STEP(H, d, a, b, c, GET(8), 0x8771f681, 11);
                STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16);
                STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23);
                STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4);
                STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11);
                STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16);
                STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23);
                STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4);
                STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11);
                STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16);
                STEP(H, b, c, d, a, GET(6), 0x04881d05, 23);
                STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4);
                STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11);
                STEP(H, c, d, a, b, 0, 0x1fa27cf8, 16);
                STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23);

                /* Round 4 */
                STEP(I, a, b, c, d, GET(0), 0xf4292244, 6);
                STEP(I, d, a, b, c, GET(7), 0x432aff97, 10);
                STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15);
                STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21);
                STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6);
                STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10);
                STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15);
                STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21);
                STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6);
                STEP(I, d, a, b, c, 0, 0xfe2ce6e0, 10);
                STEP(I, c, d, a, b, GET(6), 0xa3014314, 15);
                STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21);
                STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6);

/* We should reverse this and many rounds above instead */
                a += 0x67452301;

                // Compare the hashes and return the matched count
                if (loaded_count != 0) {
                    uint hash;

                    if (hash_num == MD5_PASSWORD_HASH_SIZE_2) {
                        hash = get_hash_high(a);
                    }
                    else {
                        hash = get_hash_low(a);
                    }

                    int val[4] = {0};
                    val[0] = bitmaps_val(a, use_local, bitmaps, local_bitmaps, 0, bitmaps_num);
                    if (val[0]) {
                        bool have_b = 0;
                        if (!have_b) {
                            a -= 0x67452301;
                            STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10);
                            STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15);
                            STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21);
                            a += 0x67452301;
                            b += 0xefcdab89;
                            c += 0x98badcfe;
                            d += 0x10325476;
                            have_b = 1;

                            val[1] = bitmaps_val(b, use_local, bitmaps, local_bitmaps, 1, bitmaps_num);
                            if (val[1]) {
                            val[2] = bitmaps_val(c, use_local, bitmaps, local_bitmaps, 2, bitmaps_num);
                            if (val[2]) {
                            val[3] = bitmaps_val(d, use_local, bitmaps, local_bitmaps, 3, bitmaps_num);
                            if (val[3]) {
                            if (use_local)
                                hash = get_hash_low(d);
                            else
                                hash = get_hash_high(d);
                                
                                int hash_index = hashtable[hash >> MD5_HASH_SHR];
                                if ( hash_index != -1)
                                    do {
                                        if ( a == loaded_hash[hash_index] && b == loaded_hash[hash_index+loaded_count] && c == loaded_hash[hash_index+loaded_count*2] && d == loaded_hash[hash_index+loaded_count*3] ){
                                        uint index = atom_inc(matched_count);
                                        uint m_base  = index * KEY_LENGTH;

                                        PUTCHAR(key, length, '\0');
                                        char *q = (char*)key;

                                        for (int j = 0; j <= length; ++j)
                                            PUTCHAR(matched_keys, m_base+j, q[j]);

                                        hashes[index] = a;
                                        hashes[loaded_count + index] = b;
                                        hashes[2 * loaded_count + index] = c;
                                        hashes[3 * loaded_count + index] = d;
                                        }
                                    } while ((hash_index = loaded_next_hash[hash_index])!=-1);
                            }}}
                        }
                    }
                }
                else {
                    a -= 0x67452301;
                    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10);
                    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15);
                    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21);
                    a += 0x67452301;
                    b += 0xefcdab89;
                    c += 0x98badcfe;
                    d += 0x10325476;
                    hashes[id] = a;
                    hashes[1 * num_keys + id] = b;
                    hashes[2 * num_keys + id] = c;
                    hashes[3 * num_keys + id] = d;
                }
            }

        } // End for alpha_j
    } // End for alpha_i
}
