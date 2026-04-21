<script setup lang="ts">
defineProps<{
  label: string
  padding?: string
}>()
</script>

<template>
  <div class="teal-callout" :style="padding ? { padding } : {}">
    <div class="mono-label text-teal-700">{{ label }}</div>
    <slot />
  </div>
</template>
