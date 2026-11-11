/* Process-owned H2D staging with persistent host DMA registration.
 * CPU contents are refreshed for every batch. Events protect asynchronous
 * readers before a slot can be overwritten. Original RDMA pools stay intact. */
#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <ascend_hal.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SLOT_BYTES (128UL * 1024 * 1024)
#define SLOT_COUNT 2
struct Slot { void *allocation, *host; aclrtEvent event; int armed; };
static struct Slot slots[SLOT_COUNT];
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static int registered_device = -1, initialized, log_fd = -1;
static unsigned long sequence;

static uint64_t now_ns(void)
{
    struct timespec stamp; clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (uint64_t)stamp.tv_sec * 1000000000ULL + stamp.tv_nsec;
}

static void *resolve(const char *name)
{
    void *address = dlsym(RTLD_NEXT, name);
    if (!address) _Exit(125);
    return address;
}

__attribute__((constructor)) static void open_log(void)
{
    const char *path = getenv("DMA_STAGE_LOG");
    if (path) log_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0600);
}

static void initialization_log(uint64_t begin, int error)
{
    if (log_fd < 0) return;
    char line[500];
    int length = snprintf(line, sizeof(line),
        "{\"event\":\"dma_stage_initialize\",\"start_ns\":%llu,\"end_ns\":%llu,\"device\":%d,\"slots\":%d,\"slot_bytes\":%zu,\"flag\":%d,\"error\":%d}\n",
        (unsigned long long)begin, (unsigned long long)now_ns(), registered_device,
        SLOT_COUNT, (size_t)SLOT_BYTES, HOST_MEM_MAP_DMA, error);
    write(log_fd, line, length);
}

static int initialize(int device)
{
    if (initialized) return registered_device == device ? 0 : 100000;
    typedef aclError (*Allocate)(void **, size_t);
    typedef drvError_t (*Register)(void *, UINT64, UINT32, UINT32, void **);
    uint64_t begin = now_ns(); registered_device = device;
    for (int i = 0; i < SLOT_COUNT; i++) {
        int error = ((Allocate)resolve("aclrtMallocHost"))(&slots[i].allocation, SLOT_BYTES + 4096);
        if (error) { initialization_log(begin, error); return error; }
        slots[i].host = (void *)(((uintptr_t)slots[i].allocation + 4095) & ~(uintptr_t)4095);
        void *unused = NULL;
        error = ((Register)resolve("halHostRegister"))(slots[i].host, SLOT_BYTES,
                    HOST_MEM_MAP_DMA, device, &unused);
        if (error) { initialization_log(begin, error); return error; }
    }
    initialized = 1; initialization_log(begin, 0); return 0;
}

aclError aclrtMemcpyBatchAsync(void **dst, size_t *maximum, void **src, size_t *sizes,
    size_t count, aclrtMemcpyBatchAttr *attr, size_t *indices, size_t nattr,
    size_t *failed, aclrtStream stream)
{
    typedef aclError (*Batch)(void **, size_t *, void **, size_t *, size_t,
        aclrtMemcpyBatchAttr *, size_t *, size_t, size_t *, aclrtStream);
    Batch batch = (Batch)resolve("aclrtMemcpyBatchAsync");
    int h2d = attr && indices && nattr == 1 && indices[0] == 0 &&
        attr[0].srcLoc.type == ACL_MEM_LOCATION_TYPE_HOST &&
        attr[0].dstLoc.type == ACL_MEM_LOCATION_TYPE_DEVICE;
    if (!h2d) return batch(dst, maximum, src, sizes, count, attr, indices, nattr, failed, stream);
    size_t bytes = 0;
    for (size_t i = 0; i < count; i++) {
        if (sizes[i] > SLOT_BYTES - bytes) return 100000;
        bytes += sizes[i];
    }
    if (!count) return batch(dst, maximum, src, sizes, count, attr, indices, nattr, failed, stream);
    pthread_mutex_lock(&gate);
    uint64_t begin = now_ns();
    typedef aclError (*GetDevice)(int32_t *);
    int32_t device = -1;
    int error = ((GetDevice)resolve("aclrtGetDevice"))(&device);
    if (!error) error = initialize(device);
    if (error) { pthread_mutex_unlock(&gate); return error; }
    unsigned long current = sequence++; int slot_id = current % SLOT_COUNT;
    struct Slot *slot = &slots[slot_id];
    typedef aclError (*EventOp)(aclrtEvent);
    typedef aclError (*CreateEvent)(aclrtEvent *);
    typedef aclError (*RecordEvent)(aclrtEvent, aclrtStream);
    uint64_t wait_begin = now_ns();
    if (slot->armed) {
        error = ((EventOp)resolve("aclrtSynchronizeEvent"))(slot->event);
        if (!error) error = ((EventOp)resolve("aclrtDestroyEvent"))(slot->event);
        if (error) { pthread_mutex_unlock(&gate); return error; }
        slot->armed = 0;
    }
    error = ((CreateEvent)resolve("aclrtCreateEvent"))(&slot->event);
    uint64_t copy_begin = now_ns();
    if (error) { pthread_mutex_unlock(&gate); return error; }
    void **redirect = malloc(count * sizeof(*redirect));
    if (!redirect) { pthread_mutex_unlock(&gate); return 100000; }
    size_t offset = 0;
    for (size_t i = 0; i < count; i++) {
        redirect[i] = (char *)slot->host + offset;
        memcpy(redirect[i], src[i], sizes[i]); offset += sizes[i];
    }
    uint64_t submit_begin = now_ns();
    error = batch(dst, maximum, redirect, sizes, count, attr, indices, nattr, failed, stream);
    uint64_t submit_end = now_ns();
    if (!error) error = ((RecordEvent)resolve("aclrtRecordEvent"))(slot->event, stream);
    if (!error) slot->armed = 1;
    if (log_fd >= 0) {
        char line[850];
        int length = snprintf(line, sizeof(line),
            "{\"event\":\"dma_stage_h2d\",\"start_ns\":%llu,\"end_ns\":%llu,\"device\":%d,\"sequence\":%lu,\"slot\":%d,\"stream\":\"%p\",\"copies\":%zu,\"bytes\":%zu,\"slot_wait_and_event_ns\":%llu,\"cpu_copy_ns\":%llu,\"batch_submit_ns\":%llu,\"src_first\":\"%p\",\"staging_first\":\"%p\",\"dst_first\":\"%p\",\"error\":%d}\n",
            (unsigned long long)begin, (unsigned long long)now_ns(), device, current,
            slot_id, stream, count, bytes, (unsigned long long)(copy_begin - wait_begin),
            (unsigned long long)(submit_begin - copy_begin),
            (unsigned long long)(submit_end - submit_begin), src[0], redirect[0], dst[0], error);
        write(log_fd, line, length);
    }
    free(redirect); pthread_mutex_unlock(&gate); return error;
}
