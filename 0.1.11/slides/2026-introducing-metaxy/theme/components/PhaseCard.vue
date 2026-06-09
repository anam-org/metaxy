<script setup lang="ts">
defineProps<{
  phase: string
  title: string
  output: string
  variant?: 'solid' | 'dashed'
  padding?: string
}>()
</script>

<template>
  <div
    :class="variant === 'dashed' ? 'phase-card-dashed' : 'phase-card-solid'"
    :style="padding ? { padding } : {}"
  >
    <div>
      <div class="mono-label text-teal-700">{{ phase }}</div>
      <h3 class="j-serif mt-1.5 text-2xl text-neutral-950">{{ title }}</h3>
    </div>
    <ul class="flex-1 space-y-1 mt-3">
      <slot />
    </ul>
    <div class="border-t border-neutral-300 pt-2 mt-2">
      <span class="mono-label text-neutral-500">Output: </span>
      <span class="text-sm text-neutral-700">{{ output }}</span>
    </div>
  </div>
</template>
